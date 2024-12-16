#pragma once

#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

__global__ void phase_encoding(
    kmm::NDRange range,
    gpu_subview_mut<cfloat, 2> ph_en_echos,
    gpu_subview<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.begin(0));
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.begin(1));

    if (voxel < range.end(0) && readout < range.end(1)) {
        auto k_y = trajectory.k_start[readout].imag();
        auto y = parameters.get(voxel).y;

        ph_en_echos[readout][voxel] = echos[readout][voxel] * exp(cfloat(0.0F, y * k_y));

        if (voxel+1 == 25414 && readout+1 == 782) {
            printf("%d %d] %p %p | %f %f -> %f %f | %f %f\n",
                   voxel,
                   readout,
                   echos.data(),
                   ph_en_echos.data(),
                   echos[readout][voxel].real(),
                   echos[readout][voxel].imag(),
                   ph_en_echos[readout][voxel].real(),
                   ph_en_echos[readout][voxel].imag(),
                   k_y,
                   y
                   );
        }
    }
}

}  // namespace kernels
}  // namespace compas
