#include "compas/parameters/tissue.h"

#include "compas/core/utils.h"

namespace compas {
TissueParameters make_tissue_parameters(
    const CompasContext& ctx,
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
    auto stride = round_up_to_multiple_of(num_voxels, 32);
    auto params = kmm::Array<float, 2> {{TissueParameterField::NUM_FIELDS, stride}};

    bool has_z = !z.is_empty();
    bool has_b0 = !B0.is_empty();
    bool has_b1 = !B1.is_empty();

    ctx.submit_device(
        num_voxels,
        [&](kmm::DeviceContext& device, kmm::NDRange, cuda_view_mut<float, 2> params) {
            device.fill(params, 0.0F);

            device.copy(
                T1.data(),
                params.drop_axis<0>(TissueParameterField::T1).data(),
                num_voxels);
            device.copy(
                T2.data(),
                params.drop_axis<0>(TissueParameterField::T2).data(),
                num_voxels);

            device.copy(
                rho_x.data(),
                params.drop_axis<0>(TissueParameterField::RHO_X).data(),
                num_voxels);
            device.copy(
                rho_y.data(),
                params.drop_axis<0>(TissueParameterField::RHO_Y).data(),
                num_voxels);

            device.copy(x.data(), params.drop_axis<0>(TissueParameterField::X).data(), num_voxels);
            device.copy(y.data(), params.drop_axis<0>(TissueParameterField::Y).data(), num_voxels);

            if (has_z) {
                device.copy(
                    z.data(),
                    params.drop_axis<0>(TissueParameterField::Z).data(),
                    num_voxels);
            }

            if (has_b0) {
                device.copy(
                    B0.data(),
                    params.drop_axis<0>(TissueParameterField::B0).data(),
                    num_voxels);
            }

            if (has_b1) {
                device.copy(
                    B1.data(),
                    params.drop_axis<0>(TissueParameterField::B1).data(),
                    num_voxels);
            } else {
                // The default value for B1 is 1
                device.fill(params.drop_axis<0>(TissueParameterField::B1), 1.0f);
            }
        },
        write(params));

    params.synchronize();

    return {
        params,
        num_voxels,
        has_z,
        has_b0,
        has_b1,
    };
}
}  // namespace compas
