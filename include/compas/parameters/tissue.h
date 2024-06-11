#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/core/object.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

/**
 * Stores the tissue parameters for each voxel. Note that, instead of having seperate 1D arrays for each field, data
 * is instead stored in a single 2D matrix where each row is different field (see `TissueParameterField`) and each
 * column is a voxel.
 */
struct TissueParameters: public Object {
    Array<float, 2> parameters;  // Size: [TissueParameterField::NUM_FIELDS, nvoxels]
    int nvoxels;
    bool has_z = true;
    bool has_b0 = true;
    bool has_b1 = true;

    TissueParameters(
        Array<float, 2> parameters,
        int nvoxels,
        bool has_z,
        bool has_b0,
        bool has_b1) :
        parameters(parameters),
        nvoxels(nvoxels),
        has_z(has_z),
        has_b0(has_b0),
        has_b1(has_b1) {}
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

namespace kmm {
template<>
struct TaskArgument<ExecutionSpace::Cuda, compas::TissueParameters> {
    using type = compas::TissueParametersView;

    static TaskArgument pack(TaskBuilder& builder, compas::TissueParameters p) {
        return {
            {.parameters = {},  //
             .nvoxels = p.nvoxels,
             .has_z = p.has_z,
             .has_b0 = p.has_b0,
             .has_b1 = p.has_b1},
            pack_argument<ExecutionSpace::Cuda>(builder, p.parameters)};
    }

    type unpack(TaskContext& context) {
        view.parameters = unpack_argument<ExecutionSpace::Cuda>(context, params);
        return view;
    }

    compas::TissueParametersView view;
    PackedArray<const float, 2> params;
};
}  // namespace kmm