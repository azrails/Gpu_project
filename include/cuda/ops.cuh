#pragma once

void cuda_set_element_impl(float *data, size_t idx, float value);
float cuda_get_element_impl(float *data, size_t idx, size_t size);

void cuda_add_impl(float* src, float* other, float* res, int size);
void cuda_add_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, int size
);

void cuda_mul_impl(float* src, float* other, float* res, int size);
void cuda_mul_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
);

void cuda_div_impl(float* src, float* other, float* res, int size);
void cuda_div_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
);

void cuda_sum_impl(
    float* src, float* res, const int *src_shape, int dim, int ndim
);
void cuda_sum_backward_impl(
    float* src_grad, float* res_grad, float* src_data, const int* src_shape, int dim, int ndim
);

//Future
// void cuda_matmul_impl(float* src1, float* src2, float* dst, int size);
// void cuda_transpose_impl(float* src1, float* dst, int size);
// void cuda_transpose_backward_impl(float* src1, float* dst, int size);