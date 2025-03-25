#include "compas/parameters/tissue.h"

#include "compas/core/utils.h"

namespace compas {
TissueParameters make_tissue_parameters(
    const CompasContext& ctx,
    int num_voxels,
    int chunk_size,
    View<float> T1,
    View<float> T2,
    View<float> B1,
    View<float> B0,
    View<float> rho_x,
    View<float> rho_y,
    View<float> x,
    View<float> y,
    View<float> z) {
    using namespace kmm::placeholders;
    auto params = kmm::Array<float, 2> {{TissueParameterField::NUM_FIELDS, num_voxels}};

    bool has_z = !z.is_empty();
    bool has_b0 = !B0.is_empty();
    bool has_b1 = !B1.is_empty();

    ctx.parallel_device(
        num_voxels,
        chunk_size,
        [&](kmm::DeviceResource& device,
            kmm::Range<index_t> range,
            GPUSubviewMut<float, 2> params) {
            KMM_ASSERT(params.is_contiguous());
            device.fill(params.data(), params.size(), 0.0F);
            auto offset = range.begin;
            auto length = range.size();

            device.copy(
                T1.data_at(offset),
                params.data_at(TissueParameterField::T1, offset),
                length);
            device.copy(
                T2.data_at(offset),
                params.data_at(TissueParameterField::T2, offset),
                length);

            device.copy(
                rho_x.data_at(offset),
                params.data_at(TissueParameterField::RHO_X, offset),
                length);
            device.copy(
                rho_y.data_at(offset),
                params.data_at(TissueParameterField::RHO_Y, offset),
                length);

            device.copy(x.data_at(offset), params.data_at(TissueParameterField::X, offset), length);
            device.copy(y.data_at(offset), params.data_at(TissueParameterField::Y, offset), length);

            if (has_z) {
                device.copy(
                    z.data_at(offset),
                    params.data_at(TissueParameterField::Z, offset),
                    length);
            }

            if (has_b0) {
                device.copy(
                    B0.data_at(offset),
                    params.data_at(TissueParameterField::B0, offset),
                    length);
            }

            if (has_b1) {
                device.copy(
                    B1.data_at(offset),
                    params.data_at(TissueParameterField::B1, offset),
                    length);
            } else {
                // The default value for B1 is 1
                device.fill(params.data_at(TissueParameterField::B1, offset), length, 1.0F);
            }
        },
        _i,
        write(params(_, _i)));

    params.synchronize();

    return {
        params,
        has_z,
        has_b0,
        has_b1,
    };
}
}  // namespace compas
