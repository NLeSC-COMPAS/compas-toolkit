#pragma once

#include "core/all.h"

namespace compas {

struct Voxel {
    float T_1;
    float T_2;
    float B_1;
    float B_0;
    cfloat rho;
    float x;
    float y;
    float z;
};

struct TissueParameters {
    CudaView<const float> T_1;
    CudaView<const float> T_2;
    CudaView<const float> B_0;
    CudaView<const cfloat> rho;
    CudaView<const float> x;
    CudaView<const float> y;

    COMPAS_HOST_DEVICE Voxel get_voxel(int voxel_idx) const {
        return Voxel {
            .T_1 = T_1[voxel_idx],
            .T_2 = T_2[voxel_idx],
            .B_1 = 1,
            .B_0 = B_0[voxel_idx],
            .rho = rho[voxel_idx],
            .x = x[voxel_idx],
            .y = y[voxel_idx],
            .z = 0};
    }
};

}  // namespace compas