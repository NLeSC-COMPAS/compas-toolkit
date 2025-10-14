#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/utils/gemm.h"
#include "gemm_kernels.cuh"

namespace compas {

template<typename T>
void compute_gemm_impl(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 2> result,
    GPUSubview<T, 2> lhs,
    GPUSubview<T, 2> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    int64_t m = result.size(0);
    int64_t n = result.size(1);
    int64_t k = lhs.size(1);

    COMPAS_CHECK(result.size(0) == m);
    COMPAS_CHECK(result.size(1) == n);
    COMPAS_CHECK(lhs.size(0) == m);
    COMPAS_CHECK(lhs.size(1) == k);
    COMPAS_CHECK(rhs.size(0) == n);
    COMPAS_CHECK(rhs.size(1) == k);

#if defined(COMPAS_USE_CUDA)
    cublasGemmAlgo_t compute_algo = CUBLAS_GEMM_DEFAULT;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType input_type = CUDA_R_32F;
    cudaDataType output_type = CUDA_R_32F;

    if constexpr (std::is_same_v<T, float>) {
        switch (kind) {
            case GemmComputeMethod::Pedantic:
                compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
                break;
            case GemmComputeMethod::Regular:
                compute_type = CUBLAS_COMPUTE_32F;
                break;
            case GemmComputeMethod::Fast:
                compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
                break;
        }
    } else if constexpr (std::is_same_v<T, kernel_float::bfloat16_t>) {
        input_type = CUDA_R_16BF;

        switch (kind) {
            case GemmComputeMethod::Pedantic:
                compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
                break;
            case GemmComputeMethod::Regular:
            case GemmComputeMethod::Fast:
                compute_type = CUBLAS_COMPUTE_32F;
                break;
        }
    } else {
        COMPAS_ERROR("invalid data type for GEMM");
    }

    COMPAS_GPU_CHECK(cublasSetStream(context.blas(), context.stream()));
    COMPAS_GPU_CHECK(cublasGemmEx_64(
        context.blas(),
        CUBLAS_OP_T,  // transa
        CUBLAS_OP_N,  // transb
        n,  // m
        m,  // n
        k,  // k
        &alpha,  // alpha
        rhs.data(),  // A
        input_type,  // A type
        rhs.stride(),  // lda
        lhs.data(),  // B
        input_type,  // B type
        lhs.stride(),  // ldb
        &beta,  //beta
        result.data(),  // C
        output_type,  // C type
        result.stride(),  // ldc
        compute_type,
        compute_algo));
#elif defined(COMPAS_USE_HIP)
    rocblas_gemm_algo compute_algo = rocblas_gemm_algo_standard;
    rocblas_datatype output_type = rocblas_datatype_f32_r;
    rocblas_datatype input_type;

    if constexpr (std::is_same_v<T, float>) {
        input_type = rocblas_datatype_f32_r;
    } else if constexpr (std::is_same_v<T, kernel_float::bfloat16_t>) {
        input_type = rocblas_datatype_bf16_r;
    } else {
        COMPAS_ERROR("invalid data type for GEMM");
    }

    COMPAS_GPU_CHECK(rocblas_set_stream(context.blas(), context.stream()));
    COMPAS_GPU_CHECK(rocblas_gemm_ex(
        context.blas(),
        rocblas_operation_transpose,  // transa
        rocblas_operation_none,  // transb
        n,  // m
        m,  // n
        k,  // k
        &alpha,  // alpha
        rhs.data(),  // A
        input_type,  // A type
        rhs.stride(),  // lda
        lhs.data(),  // B
        input_type,  // B type
        lhs.stride(),  // ldb
        &beta,  //beta
        result.data(),  // C
        output_type,  // C type
        result.stride(),  // ldc
        result.data(),  // C
        output_type,  // C type
        result.stride(),  // ldc
        output_type,
        compute_algo,
        0,
        0));
#endif
}

void compute_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 2> result,
    GPUSubview<float, 2> lhs,
    GPUSubview<float, 2> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    compute_gemm_impl(context, result, lhs, rhs, alpha, beta, kind);
}

void compute_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 2> result,
    GPUSubview<kernel_float::bfloat16_t, 2> lhs,
    GPUSubview<kernel_float::bfloat16_t, 2> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    compute_gemm_impl(context, result, lhs, rhs, alpha, beta, kind);
}

template<typename T>
void compute_complex_gemm_impl(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> result,
    GPUSubview<T, 3> lhs,
    GPUSubview<T, 3> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    COMPAS_CHECK(result.size(0) == 2);
    COMPAS_CHECK(lhs.size(0) == 2);
    COMPAS_CHECK(rhs.size(0) == 2);

    auto result_re = result.drop_axis(0);
    auto lhs_re = lhs.drop_axis(0);
    auto rhs_re = rhs.drop_axis(0);

    auto result_im = result.drop_axis(1);
    auto lhs_im = lhs.drop_axis(1);
    auto rhs_im = rhs.drop_axis(1);

    compute_gemm(context, result_re, lhs_re, rhs_re, alpha, beta, kind);
    compute_gemm(context, result_re, lhs_im, rhs_im, -alpha, 1.0f, kind);

    compute_gemm(context, result_im, lhs_re, rhs_im, alpha, beta, kind);
    compute_gemm(context, result_im, lhs_im, rhs_re, alpha, 1.0f, kind);
}

void compute_complex_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> result,
    GPUSubview<float, 3> lhs,
    GPUSubview<float, 3> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    compute_complex_gemm_impl(context, result, lhs, rhs, alpha, beta, kind);
}

void compute_complex_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> result,
    GPUSubview<kernel_float::bfloat16_t, 3> lhs,
    GPUSubview<kernel_float::bfloat16_t, 3> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind) {
    compute_complex_gemm_impl(context, result, lhs, rhs, alpha, beta, kind);
}

template<typename T>
void convert_complex_to_planar_impl(
    const kmm::DeviceResource& context,  //
    GPUSubviewMut<T, 3> output,
    GPUSubview<cfloat, 2> input) {
    auto n = kmm::checked_cast<index_t>(input.size(0));
    auto m = kmm::checked_cast<index_t>(input.size(1));

    COMPAS_CHECK(output.size(0) == 2);
    COMPAS_CHECK(output.size(1) == n);
    COMPAS_CHECK(output.size(2) == m);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(m), block_dim.x), div_ceil(uint(n), block_dim.y)};

    kernels::convert_complex_to_planar<<<grid_dim, block_dim, 0, context>>>(
        n,
        m,
        output.shift_to_origin(),
        input.shift_to_origin());
    COMPAS_GPU_CHECK(gpuGetLastError());
}

void convert_complex_to_planar(
    const kmm::DeviceResource& context,
    GPUSubviewMut<kernel_float::bfloat16_t, 3> output,
    GPUSubview<cfloat, 2> input) {
    convert_complex_to_planar_impl(context, output, input);
}

void convert_complex_to_planar(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> output,
    GPUSubview<cfloat, 2> input) {
    convert_complex_to_planar_impl(context, output, input);
}

void convert_planar_to_complex(
    const kmm::DeviceResource& context,  //
    GPUSubviewMut<cfloat, 2> output,
    GPUSubview<float, 3> input) {
    auto n = kmm::checked_cast<index_t>(output.size(0));
    auto m = kmm::checked_cast<index_t>(output.size(1));

    COMPAS_CHECK(input.size(0) == 2);
    COMPAS_CHECK(input.size(1) == n);
    COMPAS_CHECK(input.size(2) == m);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(m), block_dim.x), div_ceil(uint(n), block_dim.y)};

    kernels::convert_planar_to_complex<<<grid_dim, block_dim, 0, context>>>(
        n,
        m,
        output.shift_to_origin(),
        input.shift_to_origin());
    COMPAS_GPU_CHECK(gpuGetLastError());
}

}  // namespace compas
