#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int dv_attn_forward_wrapper(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
                       int num_levels, int num_heads, int num_depth, at::Tensor level_end_index_tensor, at::Tensor geom_xyz_tensor, at::Tensor input_features_tensor, at::Tensor depth_vector_tensor, 
                       at::Tensor attn_weights_tensor, at::Tensor output_features_tensor, at::Tensor pos_memo_tensor, at::Tensor grid_cnt_tensor);

void dv_attn_forward_kernel_launcher(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                                int num_levels, int num_heads, int num_depth, const long *level_end_index, const int *geom_xyz, const float *input_features, const float *depth_vector,
                                const float *attn_weights, float *output_features, int *pos_memo, int *grid_cnt, cudaStream_t stream);

int dv_attn_backward_wrapper(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                       int num_levels, int num_heads, int num_depth, at::Tensor level_end_index_tensor, at::Tensor pos_memo_tensor, at::Tensor input_features_tensor, at::Tensor depth_vector_tensor, at::Tensor attn_weights_tensor, 
                       at::Tensor grad_input_features_tensor, at::Tensor grad_depth_vector_tensor, at::Tensor grad_attn_weights_tensor, at::Tensor grid_cnt_tensor, at::Tensor grad_output_features_tensor);

void dv_attn_backward_kernel_launcher(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                                int num_levels, int num_heads, int num_depth, const long *level_end_index, const int *pos_memo, const float *input_features, const float *depth_vector, const float *attn_weights, 
                                float *grad_input_features, float *grad_depth_vector, float *grad_attn_weights, const int *grid_cnt, const float *grad_output_features, cudaStream_t stream);


int dv_attn_forward_wrapper(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
                       int num_levels, int num_heads, int num_depth, at::Tensor level_end_index_tensor, at::Tensor geom_xyz_tensor, at::Tensor input_features_tensor, at::Tensor depth_vector_tensor, 
                       at::Tensor attn_weights_tensor, at::Tensor output_features_tensor, at::Tensor pos_memo_tensor, at::Tensor grid_cnt_tensor) {
    CHECK_INPUT(level_end_index_tensor);
    CHECK_INPUT(geom_xyz_tensor);
    CHECK_INPUT(input_features_tensor);
    CHECK_INPUT(depth_vector_tensor);
    CHECK_INPUT(attn_weights_tensor);
    CHECK_INPUT(output_features_tensor);
    CHECK_INPUT(pos_memo_tensor);
    CHECK_INPUT(grid_cnt_tensor);
    long *level_end_index = level_end_index_tensor.data_ptr<long>();
    const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
    const float *input_features = input_features_tensor.data_ptr<float>();
    const float *depth_vector = depth_vector_tensor.data_ptr<float>();
    const float *attn_weights = attn_weights_tensor.data_ptr<float>();
    float *output_features = output_features_tensor.data_ptr<float>();
    int *pos_memo = pos_memo_tensor.data_ptr<int>();
    int *grid_cnt = grid_cnt_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dv_attn_forward_kernel_launcher(batch_size, num_points, num_cams, num_features, num_channels, num_voxel_x, num_voxel_y, num_voxel_z,
                                num_levels, num_heads, num_depth, level_end_index, geom_xyz, input_features, depth_vector, attn_weights,
                                output_features, pos_memo, grid_cnt, stream);
    return 1;
}

int dv_attn_backward_wrapper(int batch_size, int num_points, int num_cams, int num_features, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, 
                       int num_levels, int num_heads, int num_depth, at::Tensor level_end_index_tensor, at::Tensor pos_memo_tensor, at::Tensor input_features_tensor, at::Tensor depth_vector_tensor, at::Tensor attn_weights_tensor, 
                       at::Tensor grad_input_features_tensor, at::Tensor grad_depth_vector_tensor, at::Tensor grad_attn_weights_tensor, at::Tensor grid_cnt_tensor, at::Tensor grad_output_features_tensor) {
    CHECK_INPUT(level_end_index_tensor);
    CHECK_INPUT(pos_memo_tensor);
    CHECK_INPUT(input_features_tensor);
    CHECK_INPUT(depth_vector_tensor);
    CHECK_INPUT(attn_weights_tensor);
    CHECK_INPUT(grad_input_features_tensor);
    CHECK_INPUT(grad_depth_vector_tensor);
    CHECK_INPUT(grad_attn_weights_tensor);
    CHECK_INPUT(grid_cnt_tensor);
    CHECK_INPUT(grad_output_features_tensor);
    long *level_end_index = level_end_index_tensor.data_ptr<long>();
    const int *pos_memo = pos_memo_tensor.data_ptr<int>();
    const float *input_features = input_features_tensor.data_ptr<float>();
    const float *depth_vector = depth_vector_tensor.data_ptr<float>();
    const float *attn_weights = attn_weights_tensor.data_ptr<float>();
    float *grad_input_features = grad_input_features_tensor.data_ptr<float>();
    float *grad_depth_vector = grad_depth_vector_tensor.data_ptr<float>();
    float *grad_attn_weights = grad_attn_weights_tensor.data_ptr<float>();
    int *grid_cnt = grid_cnt_tensor.data_ptr<int>();
    const float *grad_output_features = grad_output_features_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dv_attn_backward_kernel_launcher(batch_size, num_points, num_cams, num_features, num_channels, num_voxel_x, num_voxel_y, num_voxel_z, 
                                num_levels, num_heads, num_depth, level_end_index, pos_memo, input_features, depth_vector, attn_weights, 
                                grad_input_features, grad_depth_vector, grad_attn_weights, grid_cnt, grad_output_features, stream);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dv_attn_forward_wrapper", &dv_attn_forward_wrapper, "dv_attn_forward_wrapper");
  m.def("dv_attn_backward_wrapper", &dv_attn_backward_wrapper, "dv_attn_backward_wrapper");
}
