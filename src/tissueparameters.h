#pragma once

#include "context.h"

namespace compas {

namespace TissueParameterField {
enum {
    T1 = 0,
    T2 = 1,
    B1 = 2,
    B0 = 3,
    RHO_X = 4,
    RHO_Y = 5,
    X = 6,
    Y = 7,
    Z = 8,
    NUM_FIELDS = 9,
};
}

struct TissueParameters {
    CudaArray<float, 2> parameters;
    bool has_z = true;
    bool has_b0 = true;
    bool has_b1 = true;
};

TissueParameters make_tissue_parameters(
    const CudaContext& ctx,
    int num_voxels,
    host_view<float> T1,
    host_view<float> T2,
    host_view<float> B1,
    host_view<float> B0,
    host_view<float> rho_x,
    host_view<float> rho_y,
    host_view<float> x,
    host_view<float> y,
    host_view<float> z);

}  // namespace compas