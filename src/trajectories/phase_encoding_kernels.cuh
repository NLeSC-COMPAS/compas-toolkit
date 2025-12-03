#pragma once

#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

__global__ void phase_encoding(
    kmm::Bounds<2, index_t> range,
    GPUSubviewMut<cfloat, 2> ph_en_echos,
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (voxel < range.x.end && readout < range.y.end) {
        auto k_y = trajectory.k_start[readout].imag();
        auto y = parameters.get(voxel).y;

        ph_en_echos[readout][voxel] = echos[readout][voxel] * exp(cfloat(0.0F, y * k_y));
    }
}

}  // namespace kernels
}  // namespace compas
