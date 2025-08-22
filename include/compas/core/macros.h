#pragma once

#if defined(__CUDACC__)
    #define COMPAS_IS_CUDA (1)

    #ifdef __CUDA_ARCH__
        #define COMPAS_IS_DEVICE (1)
    #else
        #define COMPAS_IS_DEVICE (0)
    #endif

    #define COMPAS_IS_HOST     (!COMPAS_IS_DEVICE)
    #define COMPAS_DEVICE      __device__ __forceinline__
    #define COMPAS_HOST_DEVICE __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
    #define COMPAS_IS_HIP (1)

    #ifdef __HIP_DEVICE_COMPILE__
        #define COMPAS_IS_DEVICE (1)
    #else
        #define COMPAS_IS_DEVICE (0)
    #endif

    #define COMPAS_IS_HOST     (!COMPAS_IS_DEVICE)
    #define COMPAS_DEVICE      __device__ inline __attribute__((always_inline))
    #define COMPAS_HOST_DEVICE __host__ __device__ inline __attribute__((always_inline))
#else
    #define COMPAS_IS_CUDA     (0)
    #define COMPAS_IS_DEVICE   (0)
    #define COMPAS_IS_HOST     (1)
    #define COMPAS_DEVICE      inline
    #define COMPAS_HOST_DEVICE inline
#endif

#if COMPAS_IS_DEVICE
    #define COMPAS_UNREACHABLE       \
        do {                         \
            __builtin_unreachable(); \
        } while (1)
    #define COMPAS_ASSUME(EXPR)               (__builtin_assume(EXPR))
    #define COMPAS_ASSUME_ALIGNED(PTR, ALIGN) (__builtin_assume_aligned(PTR, ALIGN))
#else
    #define COMPAS_UNREACHABLE \
        do {                   \
        } while (1)
    #define COMPAS_ASSUME(EXPR)               ((void)(0))
    #define COMPAS_ASSUME_ALIGNED(PTR, ALIGN) (PTR)
#endif
