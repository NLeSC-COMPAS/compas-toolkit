#include "tissue.h"

namespace compas {
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
    host_view<float> z) {
    vec2<index_t> shape = {static_cast<int>(TissueParameterField::NUM_FIELDS), num_voxels};
    CudaArray<float, 2> params = ctx.zeros<float, 2>(shape);

    params.slice(TissueParameterField::T1).copy_from(T1);
    params.slice(TissueParameterField::T2).copy_from(T2);

    params.slice(TissueParameterField::RHO_X).copy_from(rho_x);
    params.slice(TissueParameterField::RHO_Y).copy_from(rho_y);

    params.slice(TissueParameterField::X).copy_from(x);
    params.slice(TissueParameterField::Y).copy_from(y);

    bool has_z = !z.is_empty();
    if (has_z) {
        params.slice(TissueParameterField::Z).copy_from(z);
    }

    bool has_b0 = !B0.is_empty();
    if (has_b0) {
        params.slice(TissueParameterField::B0).copy_from(B0);
    }

    bool has_b1 = !B1.is_empty();
    if (has_b1) {
        params.slice(TissueParameterField::B1).copy_from(B1);
    } else {
        // The default value for B1 is "1"
        params.slice(TissueParameterField::B1).fill(1);
    }

    return {
        .parameters = params,
        .nvoxels = num_voxels,
        .has_z = has_z,
        .has_b0 = has_b0,
        .has_b1 = has_b1,
    };
}
}  // namespace compas