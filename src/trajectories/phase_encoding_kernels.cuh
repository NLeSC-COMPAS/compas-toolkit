#pragma once

#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

__global__ void phase_encoding(
    kmm::NDRange,
    gpu_view_mut<cfloat, 2> ph_en_echos,
    gpu_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    if (readout < echos.size(0) && voxel < echos.size(1)) {
        auto k_y = trajectory.k_start[readout].imag();
        auto y = parameters.get(voxel).y;

        ph_en_echos[readout][voxel] = echos[readout][voxel] * exp(cfloat(0.0f, y * k_y));
    }
}

}  // namespace kernels
}  // namespace compas
