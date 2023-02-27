import math
import warnings
from typing import Sequence
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from .dv_attention import DVAttention
from .decoder import CustomMSDeformableAttention
from .vf_self_attention import VFSelfAttention

@TRANSFORMER.register_module()
class VFTransformer(BaseModule):
    """Implements the VoxelFormer transformer.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_feature_levels=1,
                 num_cams=6,
                 T=1,
                 time_with_grad=True,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 x_bound=[-51.2, 51.2, 0.8],
                 y_bound=[-51.2, 51.2, 0.8],
                 z_bound=[-5, 3, 0.8],
                 d_bound=[2.0, 58.0, 0.5],
                 use_3d= False,
                 use_can_bus=False,
                 use_cams_embeds=True,
                 init_cfg=None,
                 **kwargs):
        super(VFTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.T = T
        self.fp16_enabled = False
        self.time_with_grad = time_with_grad

        self.use_cams_embeds = use_cams_embeds

        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.bev_x = int((self.x_bound[1] - self.x_bound[0]) / self.x_bound[2])
        self.bev_y = int((self.y_bound[1] - self.y_bound[0]) / self.y_bound[2])
        self.bev_z = int((self.z_bound[1] - self.z_bound[0]) / self.z_bound[2])
        self.use_3d = use_3d

        self.use_can_bus = use_can_bus
        if  self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
            )

        self.register_buffer('voxel_size', torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_coord', torch.Tensor([row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_num', torch.LongTensor([(row[1] - row[0]) / row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))

        self.depth_num = int((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the VFTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams*self.T, self.embed_dims))

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        for m in self.modules():
            if isinstance(m, DVAttention) or isinstance(m, CustomMSDeformableAttention) or isinstance(m, VFSelfAttention) :
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        self._is_init = True

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def get_cam2bev_geometry(self, img_feat, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        
        B, N, C, H, W = img_feat.shape
        coords_h = torch.arange(H, device=img_feat.device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feat.device).float() * pad_w / W

        index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feat.device).float()
        bin_size = (self.d_bound[1] - self.d_bound[0]) / self.depth_num
        coords_d = self.d_bound[0] + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d = coords3d.permute(0, 1, 4, 3, 2, 5).contiguous()

        return coords3d

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query,
                reference_points,
                object_query_embeds,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                depth_map=None,
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                img_metas=None,
                **kwargs):
        """Forward function for `VFTransformer`.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream network,
                each is a 5D-tensor with shape [B, N, C, H, W].
            bev_queries (Tensor): Input bev query 
                with shape [h*w, c] where c = embed_dims.
            object_query (Tensor): Input object query 
                with shape [b, num_query, c] where c = embed_dims.
            reference_points (Tensor): Input object query 
                with shape [b, num_query, 3].
            object_query_embeds (Tensor): The query embedding for decoder, 
                with shape [num_query, c].
            depth_map (tuple[Tensor]): Depth from depth net 
                with shape [bs, num_cam, depth_channel, H, W].
            bev_pos (Tensor): The bev pos used for encoder,
                with shape [bs, c, z, h, w] where c = embed_dims. 

        Returns:
            bev_embed (Tensor): Final bev query, with shape [h*w, bs, embed_dims]
            inter_states (Tensor): Results of decoder with shape [num_decoder, num_query, bs, embed_dims]
            init_reference_out (Tensor): Init reference points with shape [bs, num_query, 3]
            inter_references_out (Tensor): Reference points of decoder with shape [num_decoder, num_query, bs, 3]
        """
        assert len(mlvl_feats) == 1 #"for now only one fpn is supported"

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        feat_flatten = []
        spatial_shapes = []
        geom_xyz_list = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            geom_xyz = self.get_cam2bev_geometry(feat, img_metas)
            geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
            geom_xyz = geom_xyz.view(bs, num_cam, geom_xyz.shape[-4], geom_xyz.shape[-3] * geom_xyz.shape[-2], geom_xyz.shape[-1])
            geom_xyz_list.append(geom_xyz)

            feat = feat.flatten(3) # Left shape: (B, T * N, L, feat_h * feat_w)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds.view(1, num_cam, c, 1).to(feat.dtype)
            feat = feat + self.level_embeds[lvl].view(1, 1, c, 1).to(feat.dtype)

            if depth_map is not None:
                depth = depth_map[lvl]
                depth = depth.flatten(3) # Left shape: (B, T * N, D, feat_h * feat_w)
                depth_num = depth.shape[2]
            else:
                depth_num = 1

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, -1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_end_index = spatial_shapes.prod(1).cumsum(0)

        geom_xyz = torch.cat(geom_xyz_list, dim = -2)
        geom_xyz = geom_xyz.flatten(2, 3).contiguous()   # Left shape: (B, N, d_len * num_FPN * feat_h * feat_w, 3)

        if self.T == 2:
            prev_bev_queries = bev_queries.clone()
            if self.use_can_bus:
                # add can bus signals
                prev_can_bus = prev_bev_queries.new_tensor([img_meta['prev_can_bus'] for img_meta in img_metas])
                prev_can_bus = self.can_bus_mlp(prev_can_bus)[None, :, :]
                prev_bev_queries = prev_bev_queries + prev_can_bus

            if not self.time_with_grad:
                with torch.no_grad():
                    prev_bev = self.encoder(
                        prev_bev_queries,
                        feat_flatten[:, self.num_cams:],
                        feat_flatten[:, self.num_cams:],
                        depth[:, self.num_cams:] if depth_map else None,
                        geom_xyz[:, self.num_cams:],
                        bev_h=bev_h,
                        bev_w=bev_w,
                        bev_pos=bev_pos,
                        spatial_shapes=spatial_shapes,
                        level_end_index=level_end_index,
                        img_metas=img_metas,
                        prev_bev=None,
                        **kwargs
                    )
                    if self.use_3d:
                        prev_bev = prev_bev.view(bs, self.bev_z, self.bev_y, self.bev_x, -1)
                    else:
                        prev_bev = prev_bev.view(bs, 1, self.bev_y, self.bev_x, -1)
                    prev_bev = prev_bev.sum(dim = 1).flatten(1,2).contiguous()   # Left bev_map shape: (bev_y * bev_x, B, L)
            else:
                prev_bev = self.encoder(
                    prev_bev_queries,
                    feat_flatten[:, self.num_cams:],
                    feat_flatten[:, self.num_cams:],
                    depth[:, self.num_cams:] if depth_map else None,
                    geom_xyz[:, self.num_cams:],
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,
                    level_end_index=level_end_index,
                    img_metas=img_metas,
                    prev_bev=None,
                    **kwargs
                )
                if self.use_3d:
                    prev_bev = prev_bev.view(bs, self.bev_z, self.bev_y, self.bev_x, -1)
                else:
                    prev_bev = prev_bev.view(bs, 1, self.bev_y, self.bev_x, -1)
                prev_bev = prev_bev.sum(dim = 1).flatten(1,2).contiguous()   # Left bev_map shape: (bev_y * bev_x, B, L)
        else:
            # no time
            prev_bev = None

        if self.use_can_bus:
            # add can bus signals
            cur_can_bus = bev_queries.new_tensor([img_meta['cur_can_bus'] for img_meta in img_metas])  # [:, :]
            cur_can_bus = self.can_bus_mlp(cur_can_bus)[None, :, :]
            bev_queries = bev_queries + cur_can_bus

        assert feat_flatten.shape[1] == self.num_cams * self.T
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten[:, :self.num_cams],
            feat_flatten[:, :self.num_cams],
            depth[:, :self.num_cams] if depth_map else None,
            geom_xyz[:, :self.num_cams],
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_end_index=level_end_index,
            img_metas=img_metas,
            prev_bev=prev_bev,
            **kwargs
        )

        if self.use_3d:
            bev_embed = bev_embed.view(bs, self.bev_z, self.bev_y, self.bev_x, -1)
        else:
            bev_embed = bev_embed.view(bs, 1, self.bev_y, self.bev_x, -1)
        bev_embed = bev_embed.sum(dim = 1).flatten(1,2).contiguous()   # Left bev_map shape: (bev_y * bev_x, B, L)

        query = object_query
        query_pos = object_query_embeds
        init_reference_out = reference_points

        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
