#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "parameters/tissue_kernels.cuh"

namespace compas {

struct TissueParameters {
    CudaArray<float, 2> parameters;
    int nvoxels;
    bool has_z = true;
    bool has_b0 = true;
    bool has_b1 = true;

    TissueParametersView view() const {
        return {
            .parameters = parameters.view(),
            .nvoxels = nvoxels,
            .has_z = has_z,
            .has_b0 = has_b0,
            .has_b1 = has_b1};
    }
};

TissueParameters make_tissue_parameters(
    const CudaContext& ctx,
    int num_voxels,
    view<float> T1,
    view<float> T2,
    view<float> B1,
    view<float> B0,
    view<float> rho_x,
    view<float> rho_y,
    view<float> x,
    view<float> y,
    view<float> z);

}  // namespace compas