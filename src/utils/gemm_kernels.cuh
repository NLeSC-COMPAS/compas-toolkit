#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "kernel_float.h"

namespace compas {
namespace kernels {
template<typename T>
__global__ void
convert_complex_to_planar(int n, int m, GPUViewMut<T, 3> output, GPUView<cfloat, 2> input) {
    auto j = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto i = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (i < n && j < m) {
        output[0][i][j] = kernel_float::cast<T>(input[i][j].re);
        output[1][i][j] = kernel_float::cast<T>(input[i][j].im);
    }
}

template<typename T>
__global__ void
convert_planar_to_complex(int n, int m, GPUViewMut<cfloat, 2> output, GPUView<T, 3> input) {
    auto j = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto i = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (i < n && j < m) {
        output[i][j] = cfloat(
            kernel_float::cast<float>(input[0][i][j]),  //
            kernel_float::cast<float>(input[1][i][j]));
    }
}

}  // namespace kernels
}  // namespace compas