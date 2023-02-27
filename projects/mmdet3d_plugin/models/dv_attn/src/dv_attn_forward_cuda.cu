// Copyright (c) Megvii Inc. All rights reserved.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void dv_attn_forward_kernel(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x,
                                             int num_voxel_y, int num_voxel_z, int num_levels, int num_heads, int num_depth, const long *level_end_index, const int *geom_xyz,
                                             const float *input_features, const float *depth_vector, const float *attn_weights, float *output_features, int *pos_memo, int *grid_cnt) {
  // Each thread process only one channel of one voxel.
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= batch_size * num_points) {
    return;
  } else {
    int batch_idx = pt_idx / num_points;
    int cam_idx = pt_idx / (num_points / num_cams) % num_cams;

    int tmp_idx = pt_idx % (num_points / num_cams);
    int depth_idx = tmp_idx / num_features;
    int feature_idx = tmp_idx % num_features;

    int level_idx = num_levels;
    for (int level = 0; level < num_levels; level++) {
      if (feature_idx < level_end_index[level]) {
        level_idx = level;
        break;
      }
    }
    if (level_idx == num_levels) {
      return;
    }
    int x = geom_xyz[pt_idx * 3];
    int y = geom_xyz[pt_idx * 3 + 1];
    int z = geom_xyz[pt_idx * 3 + 2];
    // if coord of current voxel is out of boundary, return.

    if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 || z >= num_voxel_z) {
      return;
    }
    pos_memo[pt_idx * 3] = z;
    pos_memo[pt_idx * 3 + 1] = y;
    pos_memo[pt_idx * 3 + 2] = x;

    int grid_id = batch_idx * num_voxel_z * num_voxel_y * num_voxel_x + z * num_voxel_y * num_voxel_x + y * num_voxel_x + x;

    atomicAdd(&grid_cnt[grid_id], 1);

    for (int head_idx = 0; head_idx < num_heads; head_idx++)
    {
      for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        atomicAdd(
            &output_features[grid_id * num_channels * num_heads + head_idx * num_channels + channel_idx],
            input_features[(batch_idx * num_cams * num_channels * num_heads + cam_idx * num_channels * num_heads + head_idx * num_channels + channel_idx) * num_features + feature_idx] * 
              depth_vector[(batch_idx * num_cams * num_depth + cam_idx * num_depth + depth_idx) * num_features + feature_idx] *
              attn_weights[(batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_cams * num_levels * num_voxel_z * num_heads +
              cam_idx * num_levels * num_voxel_z * num_heads + level_idx * num_voxel_z * num_heads + z * num_heads + head_idx]);
      }
    }
  }
}

void dv_attn_forward_kernel_launcher(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x,
                                           int num_voxel_y, int num_voxel_z, int num_levels, int num_heads, int num_depth, const long *level_end_index, const int *geom_xyz,
                                           const float *input_features, const float *depth_vector, const float *attn_weights, float *output_features, int *pos_memo, int *grid_cnt,
                                           cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  dv_attn_forward_kernel<<<blocks, threads, 0, stream>>>(batch_size, num_points, num_cams, num_features, num_channels, num_voxel_x,
                                                               num_voxel_y, num_voxel_z, num_levels, num_heads, num_depth, level_end_index, geom_xyz, input_features, depth_vector,
                                                               attn_weights, output_features, pos_memo, grid_cnt);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void dv_attn_backward_kernel(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                                            int num_levels, int num_heads, int num_depth, const long * level_end_index, const int *pos_memo, const float *input_features, const float *depth_vector, const float *attn_weights, 
                                            float *grad_input_features, float *grad_depth_vector, float *grad_attn_weights, const int *grid_cnt, const float *grad_output_features) {
  // Each thread process only one channel of one voxel.
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= batch_size * num_points) {
    return;
  } else {
    int batch_idx = pt_idx / num_points;
    int cam_idx = pt_idx / (num_points / num_cams) % num_cams;

    int tmp_idx = pt_idx % (num_points / num_cams);
    int depth_idx = tmp_idx / num_features;
    int feature_idx = tmp_idx % num_features;

    int level_idx = num_levels;
    for (int level = 0; level < num_levels; level++) {
      if (feature_idx < level_end_index[level]) {
        level_idx = level;
        break;
      }
    }
    if (level_idx == num_levels) {
      return;
    }
    int z = pos_memo[pt_idx * 3];
    int y = pos_memo[pt_idx * 3 + 1];
    int x = pos_memo[pt_idx * 3 + 2];
    // if point is not used, return.
    if (z == -1) {
      return;
    }

    int grid_id = batch_idx * num_voxel_z * num_voxel_y * num_voxel_x + z * num_voxel_y * num_voxel_x + y * num_voxel_x + x;

    for (int head_idx = 0; head_idx < num_heads; head_idx++)
    {
      for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {

        atomicAdd(
          &grad_input_features[(batch_idx * num_cams * num_channels * num_heads + cam_idx * num_channels * num_heads + head_idx * num_channels + channel_idx) * num_features + feature_idx],
            grad_output_features[grid_id * num_channels * num_heads + head_idx * num_channels + channel_idx] * 
            depth_vector[(batch_idx * num_cams * num_depth + cam_idx * num_depth + depth_idx) * num_features + feature_idx] *
            attn_weights[(batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_cams * num_levels * num_voxel_z * num_heads +
            cam_idx * num_levels * num_voxel_z * num_heads + level_idx * num_voxel_z * num_heads + z * num_heads + head_idx] / grid_cnt[grid_id]);

        atomicAdd(
          &grad_attn_weights[(batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_cams * num_levels * num_voxel_z * num_heads +
            cam_idx * num_levels * num_voxel_z * num_heads + level_idx * num_voxel_z * num_heads + z * num_heads + head_idx],
            grad_output_features[grid_id * num_channels * num_heads + head_idx * num_channels + channel_idx] * 
            depth_vector[(batch_idx * num_cams * num_depth + cam_idx * num_depth + depth_idx) * num_features + feature_idx] *
            input_features[(batch_idx * num_cams * num_channels * num_heads + cam_idx * num_channels * num_heads + head_idx * num_channels + channel_idx) * num_features + feature_idx] / grid_cnt[grid_id]);

        atomicAdd(
          &grad_depth_vector[(batch_idx * num_cams * num_depth + cam_idx * num_depth + depth_idx) * num_features + feature_idx],
            grad_output_features[grid_id * num_channels * num_heads + head_idx * num_channels + channel_idx] * 
            input_features[(batch_idx * num_cams * num_channels * num_heads + cam_idx * num_channels * num_heads + head_idx * num_channels + channel_idx) * num_features + feature_idx] *
            attn_weights[(batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_cams * num_levels * num_voxel_z * num_heads +
            cam_idx * num_levels * num_voxel_z * num_heads + level_idx * num_voxel_z * num_heads + z * num_heads + head_idx] / grid_cnt[grid_id]);

      }
    }
  }
}

void dv_attn_backward_kernel_launcher(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                                int num_levels, int num_heads, int num_depth, const long *level_end_index, const int *pos_memo, const float *input_features, const float *depth_vector, const float *attn_weights, 
                                float *grad_input_features, float *grad_depth_vector, float *grad_attn_weights, const int *grid_cnt, const float *grad_output_features, cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  dv_attn_backward_kernel<<<blocks, threads, 0, stream>>>(batch_size, num_points, num_cams, num_features, num_channels, num_voxel_x,
                                                               num_voxel_y, num_voxel_z, num_levels, num_heads, num_depth, level_end_index, pos_memo, input_features, depth_vector, 
                                                               attn_weights, grad_input_features, grad_depth_vector, grad_attn_weights, grid_cnt, grad_output_features);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}