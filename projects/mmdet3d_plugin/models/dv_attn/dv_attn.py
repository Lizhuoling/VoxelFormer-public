import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import dv_attn_ext

class DVAttn(Function):
    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor, depth_vector: torch.Tensor, attn_weight: torch.Tensor,
                voxel_num: torch.Tensor, level_end_index: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Forward function for `Dual-View Attn.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of (B, N * d_len * feat_h * feat_w, 3)
            input_features (Tensor): feature for each voxel with the
                shape of (B, N, C, feat_h * feat_w)
            depth_vector (Tensor): feature for each voxel with the
                shape of (B, N, depth_channel, feat_h * feat_w)
            attn_weight (Tensor): 
                shape of (B, bev_h * bev_w, N * num_head * bev_z)
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of (3).
            level_end_index (Tensor):
                shape of (feat_h * feat_w)
            num_heads (int):

        Returns:
            output_features: (B, bev_z, bev_y, bev_x, C) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        assert depth_vector.is_contiguous()
        assert attn_weight.is_contiguous()
        assert voxel_num.is_contiguous()
        assert level_end_index.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        grad_input_features = torch.zeros_like(input_features)
        grad_depth_vector = torch.zeros_like(depth_vector)
        grad_attn_weights = torch.zeros_like(attn_weight)

        batch_size = input_features.shape[0]
        num_points = geom_xyz.shape[1]
        num_cams = input_features.shape[1]
        num_features = input_features.shape[3]
        num_channels = input_features.shape[2]
        num_depth = depth_vector.shape[2]
        num_levels = level_end_index.shape[-1]

        output_features = input_features.new_zeros(batch_size, voxel_num[2], voxel_num[1], voxel_num[0], num_channels)
        grid_cnt = torch.zeros((batch_size, voxel_num[2], voxel_num[1], voxel_num[0]), dtype = torch.int32).to(output_features.device)

        assert voxel_num[2] * num_cams * num_levels * num_heads == attn_weight.shape[-1]
        num_channels = num_channels // num_heads
        # # Save the position of bev_feature_map for each input point.
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        dv_attn_ext.dv_attn_forward_wrapper(
            batch_size,
            num_points,
            num_cams,
            num_features,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            num_levels,
            num_heads,
            num_depth,
            level_end_index,
            geom_xyz,
            input_features,
            depth_vector,
            attn_weight,
            output_features,
            pos_memo,
            grid_cnt,
        )

        grid_cnt[grid_cnt==0] = 1   # Avoid dividing 0
        output_features = output_features / grid_cnt.unsqueeze(-1)  # Left shape: (B, bev_z, bev_y, bev_x, C)

        ctx.num_heads = num_heads
        # save grad_input_features and pos_memo for backward
        ctx.save_for_backward(grad_input_features, grad_depth_vector, grad_attn_weights, input_features, depth_vector, attn_weight, pos_memo, voxel_num, level_end_index, grid_cnt)
        
        return output_features 

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_features):
        '''
        Input:
            grad_output_features: shape (B, bev_z, bev_y, bev_x, C)
        '''
        if not grad_output_features.is_contiguous():
            grad_output_features = grad_output_features.contiguous()
        
        num_heads = ctx.num_heads
        grad_input_features, grad_depth_vector, grad_attn_weights, input_features, depth_vector, attn_weight, pos_memo, voxel_num, level_end_index, grid_cnt = ctx.saved_tensors
        batch_size = input_features.shape[0]
        num_points = pos_memo.shape[1]
        num_cams = input_features.shape[1]
        num_features = input_features.shape[3]
        num_channels = input_features.shape[2]
        num_depth = depth_vector.shape[2]
        num_levels = level_end_index.shape[-1]

        num_channels = num_channels // num_heads
        dv_attn_ext.dv_attn_backward_wrapper(
            batch_size,
            num_points,
            num_cams,
            num_features,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            num_levels,
            num_heads,
            num_depth,
            level_end_index,
            pos_memo,
            input_features,
            depth_vector,
            attn_weight,
            grad_input_features,
            grad_depth_vector,
            grad_attn_weights,
            grid_cnt,
            grad_output_features,
        )
        
        return None, grad_input_features, grad_depth_vector, grad_attn_weights, None, None, None

dv_attn = DVAttn.apply
