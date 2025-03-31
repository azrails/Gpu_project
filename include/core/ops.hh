#pragma once

void cpu_add_impl(float* src, float* other, float* res, int size);
void cpu_add_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, int size
);

void cpu_mul_impl(float* src, float* other, float* res, int size);
void cpu_mul_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
);

void cpu_div_impl(float* src, float* other, float* res, int size);
void cpu_div_backward_impl(
    float* src_grad, float* other_grad, float* res_grad, float* src_data, float* other_data, int size
);

void cpu_sum_impl(
    float* src, float* res, const int *src_shape, int dim, int ndim
);
void cpu_sum_backward_impl(
    float* src_grad, float* res_grad, float* src_data, const int* src_shape, int dim, int ndim
);

// void cpu_matmul_impl(float* src1, float* src2, float* dst, int size);
// void cpu_transpose_impl(float* src1, float* dst, int size);
// void cpu_transpose_backward_impl(float* src1, float* dst, int size);

