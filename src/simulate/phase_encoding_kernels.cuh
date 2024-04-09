#pragma once

#include "core/view.h"
#include "parameters/tissue_view.cuh"

namespace compas {
namespace kernels {

template<typename TrajectoryView>
__global__ void phase_encoding(
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    TrajectoryView trajectory) {
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    if (readout < echos.size(0) && voxel < echos.size(1)) {
        auto y = parameters.get(voxel).y;
        auto k = trajectory.k_start[readout].imag();

        echos[readout][voxel] = exp(cfloat(0.0f, y * k));
    }
}

} // kernels
} // compas