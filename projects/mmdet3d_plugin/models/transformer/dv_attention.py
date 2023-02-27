import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
from dv_attn import dv_attn

@ATTENTION.register_module()
class DVAttention(BaseModule):
    """An attention module used in VoxelFormer called Dual-View Attention.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_levels=1,
                 bev_h = 128,
                 bev_w = 128,
                 bev_z = 8,
                 num_cam = 6,
                 num_heads = 1,
                 dropout=0.1,
                 use_3d = False,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.register_buffer('voxel_num', torch.LongTensor([bev_h, bev_w, bev_z]))

        self.embed_dims = embed_dims

        assert num_levels == 1 # for now only one FPN level is supported 
        self.num_levels = num_levels

        self.num_cam = num_cam
        self.num_heads = num_heads
        self.use_3d = use_3d
        if self.use_3d:
            self.dv_attn_fc = nn.Linear(embed_dims, self.num_cam * self.num_levels * self.num_heads)
        else:
            self.dv_attn_fc = nn.Linear(embed_dims, self.num_cam * self.num_levels * self.bev_z * self.num_heads)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.dv_attn_fc, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @force_fp32(apply_to=('query', 'value'))
    def forward(self,
                query,
                key=None,
                value=None,
                depth=None,
                identity=None,
                query_pos=None,
                geom_xyz=None,
                spatial_shapes=None,
                level_end_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of DVAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_cam, embed_dims, h*w)`.
            value (Tensor): The value tensor with shape
                `(bs, num_cam, embed_dims, h*w)`.
            depth (Tensor): The value tensor with shape
                `(bs, num_cam, num_depth, h*w)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            geom_xyz (Tensor): Input projection matrix from cam to bev with shape
                (bs, num_cam, h*w*num_depth, 3)
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_end_index (Tensor): The end index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        B,  num_query, embed_dims = query.shape
        _, _, L, num_value = value.shape
        assert level_end_index[-1] == num_value

        geom_xyz = geom_xyz.flatten(1,2)

        attn_weight = self.dv_attn_fc(query) # Left shape: (B, bev_z * bev_y * bev_x, N * num_FPN)
        if self.use_3d:
            attn_weight = attn_weight.view(B, self.bev_z, self.bev_h, self.bev_w, self.num_cam * self.num_levels * self.num_heads).sigmoid().permute(0, 2, 3, 4, 1)\
                .reshape(B, self.bev_h * self.bev_w, self.num_cam * self.num_levels * self.bev_z * self.num_heads)
        
        output = dv_attn(geom_xyz.contiguous(), value.contiguous(), depth.contiguous(), attn_weight.contiguous(), self.voxel_num.cuda(), level_end_index.contiguous(), self.num_heads)     # Left shape: (B, bev_z, bev_y, bev_x, L)

        if self.use_3d:
            output = self.output_proj(output).view(B, self.bev_z * self.bev_h * self.bev_w, L) # Left shape: (B, bev_z, bev_y, bev_x, L)
        else:
            output = self.output_proj(output).sum(1).view(B, self.bev_h * self.bev_w, L) # Left shape: (B, bev_z, bev_y, bev_x, L)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
