#pragma once

#ifdef COMPAS_USE_CUDA
    #include <cooperative_groups.h>
    #include <cooperative_groups/reduce.h>
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#elif defined(COMPAS_USE_HIP)
    #include <hip/hip_cooperative_groups.h>
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
#endif