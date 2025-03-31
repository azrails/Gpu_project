#include<cstdlib>
#include "core/ops.hh"

void cpu_add_impl(float* src, float* other, float* res, int size){
    for (size_t i = 0; i < size; ++i)
        res[i] = src[i] + other[i];
}

void cpu_add_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, int size
){
    for (int i = 0; i < size; ++i){
        if (src_grad) src_grad[i] += res_grad[i];
        if (other_grad) other_grad[i] += res_grad[i];
    }
}

void cpu_mul_impl(float* src, float* other, float* res, int size){
    for (int i = 0; i < size; ++i)
        res[i] = src[i] * other[i];
}

void cpu_mul_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    for (int i = 0; i < size; ++i){
        const float gv = res_grad[i];
        if (src_grad) src_grad[i] += gv * other_data[i];
        if (other_grad) other_grad[i] += gv * src_data[i];
    }
}

void cpu_div_impl(float* src, float* other, float* res, int size){
    for (int i = 0; i < size; ++i)
        res[i] = src[i] / other[i];
}

void cpu_div_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
){
    for (int i = 0; i < size; ++i){
        const float gv = res_grad[i];
        const float denom = other_data[i];
        const float denom_sq = denom * denom;
        if (src_grad) src_grad[i] += gv / denom;
        if (other_grad) other_grad[i] += (-gv * src_data[i]) / denom_sq;
    }
}

void cpu_sum_impl(float* src, float* res, const int* src_shape, int dim, int ndim){
    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= src_shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) {
        inner_size *= src_shape[i];
    }
    size_t dim_size = src_shape[dim];

    for (size_t outer = 0; outer < outer_size; ++outer){
        for(size_t inner = 0; inner < inner_size; ++inner){
            float sum = 0.0f;
            for (size_t d = 0; d < dim_size; ++d){
                size_t src_idx = (outer * dim_size * inner_size) +
                    (d * inner_size) + inner;
                sum += src[src_idx];
            }
            size_t dst_idx = outer * inner_size + inner;
            res[dst_idx] = sum;
        }
    }
}


void cpu_sum_backward_impl(
    float* src_grad, float* res_grad, float* src_data, const int* src_shape, int dim, int ndim
){
    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= src_shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) {
        inner_size *= src_shape[i];
    }
    size_t dim_size = src_shape[dim];

    for (size_t outer = 0; outer < outer_size; ++outer){
        for (size_t inner = 0; inner < inner_size; ++inner){
            const float gv = res_grad[outer * inner_size + inner];

            for (size_t d = 0; d < dim_size; ++d){
                size_t src_idx = (outer * dim_size * inner_size) +
                    (d * inner_size) + inner;
                src_grad[src_idx] += gv;
            }
        }
    }
}
