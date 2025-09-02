#include "compas/core/assertion.h"
#include "compas/utils/gemm.h"

namespace compas {

template <typename T>
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

    COMPAS_CHECK(lhs.size(0) == m);
    COMPAS_CHECK(lhs.size(1) == k);
    COMPAS_CHECK(rhs.size(0) == n);
    COMPAS_CHECK(rhs.size(1) == k);

#if defined(COMPAS_USE_CUDA)
    cublasGemmAlgo_t compute_algo = CUBLAS_GEMM_DEFAULT;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    switch (kind) {
        case GemmComputeMethod::Pedantic:
            compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
            break;
        case GemmComputeMethod::Fast:
            compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
            break;
        case GemmComputeMethod::BF16:
            compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
            break;
        case GemmComputeMethod::TF32:
            compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
            break;
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
            CUDA_R_32F,  // A type
            rhs.stride(),  // lda
            lhs.data(),  // B
            CUDA_R_32F,  // B type
            lhs.stride(),  // ldb
            &beta,  //beta
            result.data(),  // C
            CUDA_R_32F,  // C type
            result.stride(),  // ldc
            compute_type,
            compute_algo));
#elif defined(COMPAS_USE_HIP)
    rocblas_gemm_algo compute_algo = rocblas_gemm_algo_standard;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    switch (kind) {
        case GemmComputeMethod::Pedantic:
            compute_type = rocblas_datatype_f32_r;
            break;
        case GemmComputeMethod::Fast:
            compute_type = rocblas_datatype_f32_r;
            break;
        case GemmComputeMethod::BF16:
            // TODO: BF16_C not currently supported
            compute_type = rocblas_datatype_f32_r;
            break;
        case GemmComputeMethod::TF32:
            // TODO: TF32_C not currently supported
            compute_type = rocblas_datatype_f32_r;
            break;
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
        rocblas_datatype_f32_r,  // A type
        rhs.stride(),  // lda
        lhs.data(),  // B
        rocblas_datatype_f32_r,  // B type
        lhs.stride(),  // ldb
        &beta,  //beta
        result.data(),  // C
        rocblas_datatype_f32_r,  // C type
        result.stride(),  // ldc
        result.data(),  // C
        rocblas_datatype_f32_r,  // C type
        result.stride(),  // ldc
        compute_type,
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

void compute_gemm(
        const kmm::DeviceResource& context,
        GPUSubviewMut<float, 2> result_re,
        GPUSubviewMut<float, 2> result_im,
        GPUSubview<float, 2> lhs_re,
        GPUSubview<float, 2> lhs_im,
        GPUSubview<float, 2> rhs_re,
        GPUSubview<float, 2> rhs_im,
        float alpha,
        float beta,
        GemmComputeMethod kind) {
    // Real part: Cre = beta * Cre + alpha * Are * Bre - alpha * Aim * Bim
    compute_gemm_impl(context, result_re, lhs_re, rhs_re,  +alpha, beta, kind);
    compute_gemm_impl(context, result_re, lhs_im, rhs_im,  -alpha, 1.0F, kind);

    // Imag part: Cim = beta * Cim + alpha * Aim * Bre - alpha * Are * Bim
    compute_gemm_impl(context, result_im, lhs_re, rhs_im,  +alpha, beta, kind);
    compute_gemm_impl(context, result_im, lhs_im, rhs_re,  +alpha, 1.0F, kind);
}

void compute_gemm(
        const kmm::DeviceResource& context,
        GPUSubviewMut<float, 2> result_re,
        GPUSubviewMut<float, 2> result_im,
        GPUSubview<kernel_float::bfloat16_t, 2> lhs_re,
        GPUSubview<kernel_float::bfloat16_t, 2> lhs_im,
        GPUSubview<kernel_float::bfloat16_t, 2> rhs_re,
        GPUSubview<kernel_float::bfloat16_t, 2> rhs_im,
        float alpha,
        float beta,
        GemmComputeMethod kind) {
    // Real part: Cre = beta * Cre + alpha * Are * Bre - alpha * Aim * Bim
    compute_gemm_impl(context, result_re, lhs_re, rhs_re,  +alpha, beta, kind);
    compute_gemm_impl(context, result_re, lhs_im, rhs_im,  -alpha, 1.0F, kind);

    // Imag part: Cim = beta * Cim + alpha * Aim * Bre - alpha * Are * Bim
    compute_gemm_impl(context, result_im, lhs_re, rhs_im,  +alpha, beta, kind);
    compute_gemm_impl(context, result_im, lhs_im, rhs_re,  +alpha, 1.0F, kind);
}

}  // namespace compas
