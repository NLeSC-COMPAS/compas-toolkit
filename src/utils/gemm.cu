#include "compas/core/assertion.h"
#include "compas/utils/gemm.h"

namespace compas {

void compute_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<cfloat, 2> result,
    GPUSubview<cfloat, 2> lhs,
    GPUSubview<cfloat, 2> rhs,
    cfloat beta,
    GemmComputeMethod kind) {
    cuComplex alpha = {1, 0};

    int64_t m = result.size(0);
    int64_t n = result.size(1);
    int64_t k = lhs.size(1);

    COMPAS_CHECK(result.size(0) == m);
    COMPAS_CHECK(result.size(1) == n);
    COMPAS_CHECK(lhs.size(0) == m);
    COMPAS_CHECK(lhs.size(1) == k);
    COMPAS_CHECK(rhs.size(0) == n);
    COMPAS_CHECK(rhs.size(1) == k);

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
        CUDA_C_32F,  // A type
        rhs.stride(),  // lda
        lhs.data(),  // B
        CUDA_C_32F,  // B type
        lhs.stride(),  // ldb
        &beta,  //beta
        result.data(),  // C
        CUDA_C_32F,  // C type
        result.stride(),  // ldc
        compute_type,
        compute_algo));
}

}  // namespace compas
