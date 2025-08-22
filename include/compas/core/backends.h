#pragma once

#if defined(COMPAS_USE_CUDA)
    #include <cooperative_groups.h>
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_runtime_api.h>
    #define deviceComplex cuComplex
#elif defined(COMPAS_USE_HIP)
    #include <hip/hip_complex.h>
    #include <hip/hip_cooperative_groups.h>
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
    #define deviceComplex hipComplex
#endif