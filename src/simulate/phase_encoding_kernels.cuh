#pragma once

#include "core/view.h"
#include "trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

template<typename TrajectoryView>
__global__ void phase_encoding(
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    TrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (readout < echos.size(0) && voxel < echos.size(1)) {
        auto y = parameters.get(voxel).y;
        auto k = trajectory.k_start[readout];

        echos[readout][voxel] = exp(cfloat(0, y * k));
    }
}

} // kernels
} // compas