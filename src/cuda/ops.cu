#include "cuda/ops.cuh"
#include <stdexcept>
#include <core/cudaException.hh>

__global__ void set_element_ker(float *data, size_t idx, float value){
    if (threadIdx.x == 0 && blockIdx.x == 0)
        data[idx] = value;
}

void cuda_set_element_impl(float *data, size_t idx, float value){
    set_element_ker<<<1, 1>>>(data, idx, value);
    cudaDeviceSynchronize();
}

float cuda_get_element_impl(float *data, size_t idx, size_t size){
    if (idx >= size){
        throw std::invalid_argument("Index out of range!");
    }
    if (data == nullptr){
        throw std::invalid_argument("Data empty!");
    }
    float result;
    cudaMemcpy(&result, data + idx, sizeof(float), cudaMemcpyDeviceToHost);
    CudaException::throwIfCUDAErrorsOccurred("Get element crash!");
    return result;
}


__global__ void add_fwd_ker(float* src1, float* src2, float* dst, int size){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < size) dst[tid] = src1[tid] + src2[tid];
}

void cuda_add_impl(float* src, float* other, float* res, int size){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    add_fwd_ker<<<grid_size, block_size>>>(src, other, res, size);
    CudaException::throwIfCUDAErrorsOccurred("Add kernel crash!");
}

__global__ void mul_fwd_ker(float* src1, float* src2, float* dst, int size){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < size) dst[tid] = src1[tid] * src2[tid];
}

void cuda_mul_impl(float* src, float* other, float* res, int size){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    mul_fwd_ker<<<grid_size, block_size>>>(src, other, res, size);
    CudaException::throwIfCUDAErrorsOccurred("Mul kernel crash!");
}

__global__ void div_fwd_ker(float* src1, float* src2, float* dst, int size){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < size) dst[tid] = src1[tid] / src2[tid];
}

void cuda_div_impl(float* src, float* other, float* res, int size){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    div_fwd_ker<<<grid_size, block_size>>>(src, other, res, size);
    CudaException::throwIfCUDAErrorsOccurred("Div kernel crash!");
}

__global__ void add_bwd_ker(float* src_grad, float* other_grad, float* res_grad, int size){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid >= size) return;

    const float gv = res_grad[tid];
    if (src_grad) atomicAdd(&src_grad[tid], gv);
    if (other_grad) atomicAdd(&other_grad[tid], gv);
}

void cuda_add_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, int size
){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    add_bwd_ker<<<grid_size, block_size>>>(src_grad, other_grad, res_grad, size);
    CudaException::throwIfCUDAErrorsOccurred("Add backward kernel crash!");
}

__global__ void mul_bwd_ker(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid >= size) return;

    const float gv = res_grad[tid];
    if (src_grad) atomicAdd(&src_grad[tid], gv * other_data[tid]);
    if (other_grad) atomicAdd(&other_grad[tid], gv * src_data[tid]);
}

void cuda_mul_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    mul_bwd_ker<<<grid_size, block_size>>>(src_grad, other_grad, res_grad, src_data, other_data, size);
    CudaException::throwIfCUDAErrorsOccurred("Mul backward kernel crash!");
}

__global__ void div_bwd_ker(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid >= size) return;

    const float gv = res_grad[tid];
    const float denom = other_data[tid];
    const float denom_sq = denom * denom;

    if (src_grad) atomicAdd(&src_grad[tid], gv / denom);
    if (other_grad) atomicAdd(&other_grad[tid], (-gv * src_data[tid]) / denom_sq);
}

void cuda_div_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    div_bwd_ker<<<grid_size, block_size>>>(src_grad, other_grad, res_grad, src_data, other_data, size);
    CudaException::throwIfCUDAErrorsOccurred("Div backward kernel crash!");
}

__global__ void sum_fwd_ker(
    float* src, float* res, size_t outer_size, size_t dim_size, size_t inner_size
) {
    extern __shared__ float sdata[];
    
    size_t res_idx = blockIdx.x;
    size_t outer = res_idx / inner_size;
    size_t inner = res_idx % inner_size;
    
    if (outer >= outer_size || inner >= inner_size) return;
    
    // Каждый поток суммирует свою часть элементов по оси dim
    float sum = 0.0f;
    for (size_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
        size_t src_idx = (outer * dim_size + d) * inner_size + inner;
        sum += src[src_idx];
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        res[res_idx] = sdata[0];
    }
}

void cuda_sum_impl(
    float* src, float* res, const int* src_shape, int dim, int ndim
) {
    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) 
        outer_size *= src_shape[i];
    
    size_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) 
        inner_size *= src_shape[i];
    
    size_t dim_size = src_shape[dim];
    size_t total_res = outer_size * inner_size;
    
    const int threads_per_block = 256;
    dim3 block(threads_per_block);
    dim3 grid((total_res + block.x - 1) / block.x);
    
    
    sum_fwd_ker<<<grid, block, threads_per_block * sizeof(float)>>>(
        src, res, outer_size, dim_size, inner_size
    );
    
    cudaDeviceSynchronize();
    CudaException::throwIfCUDAErrorsOccurred("Sum kernel crash!");
}

__global__ void sum_bwd_ker(
    float* src_grad, float* res_grad, size_t outer_size, size_t dim_size, size_t inner_size
){
    const size_t outer = blockIdx.z;
    const size_t inner = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;

    if(outer >= outer_size || inner >= inner_size || d >= dim_size) return;
    size_t src_idx = (outer * dim_size + d) * inner_size + inner;
    size_t res_idx = outer * inner_size + inner;

    atomicAdd(&src_grad[src_idx], res_grad[res_idx]);
}

void cuda_sum_backward_impl(
    float* src_grad, float* res_grad, float* src_data, const int* src_shape, int dim, int ndim
){
    size_t outer_size = 1;
    for(int i = 0; i < dim; ++i) outer_size *= src_shape[i];
    

    size_t inner_size = 1;
    for(int i = dim + 1; i < ndim; ++i) inner_size *= src_shape[i];

    size_t dim_size = src_shape[dim];

    dim3 block(16, 16);

    const int blocks_x = (dim_size + 16 - 1) / block.x;
    const int blocks_y = (inner_size + 16 - 1) / block.y;
    dim3 grid(blocks_x, blocks_y, outer_size);
    sum_bwd_ker<<<grid, block>>>(src_grad, res_grad, outer_size, dim_size, inner_size);
    cudaDeviceSynchronize();
    CudaException::throwIfCUDAErrorsOccurred("Sum backward kernel crash!");
}
