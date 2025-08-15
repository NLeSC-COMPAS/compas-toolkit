#pragma once

#ifdef COMPAS_USE_CUDA
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#elif COMPAS_USE_HIP
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
#endif