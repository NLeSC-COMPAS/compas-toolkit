#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/core/object.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

/**
 * Stores the tissue parameters for each voxel. Note that, instead of having separate 1D arrays for each field, data
 * is instead stored in a single 2D matrix where each row is different field (see `TissueParameterField`) and each
 * column is a voxel.
 */
struct TissueParameters: public Object {
    Array<float, 2> data;  // Size: [TissueParameterField::NUM_FIELDS, nvoxels]
    int nvoxels;
    int chunk_size;
    bool has_z = true;
    bool has_b0 = true;
    bool has_b1 = true;

    TissueParameters(Array<float, 2> parameters, bool has_z, bool has_b0, bool has_b1) :
        data(parameters),
        nvoxels(kmm::checked_cast<int>(parameters.size(1))),
        chunk_size(kmm::checked_cast<int>(parameters.chunk_size(1))),
        has_z(has_z),
        has_b0(has_b0),
        has_b1(has_b1) {}
};

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
    View<float> z = {});

}  // namespace compas

namespace kmm {

template<typename M>
struct Argument<Read<compas::TissueParameters, M>> {
    using type = compas::TissueParametersView;

    static Argument
    pack(TaskInstance& task, Read<compas::TissueParameters, M> access) {
        const compas::TissueParameters& params = access.argument;
        compas::TissueParametersView view;
        view.has_z = params.has_z;
        view.has_b0 = params.has_b0;
        view.has_b1 = params.has_b1;

        return {view, pack_argument(task, params.data(All(), access.mode.access_mapper))};
    }

    template<ExecutionSpace Space>
    type unpack(TaskContext& context) {
        view.parameters = unpack_argument<Space>(context, params);
        return view;
    }

    compas::TissueParametersView view;
    packed_argument_t<Read<Array<float, 2>, M>> params;
};

// Forward compas::TissueParameters& to Read<compas::TissueParameters>
template<>
struct ArgumentHandler<const compas::TissueParameters&>: //
        ArgumentHandler<Read<const compas::TissueParameters>>{
    ArgumentHandler(const compas::TissueParameters& p) :
        ArgumentHandler<Read<const compas::TissueParameters>>(p) {}
};

}  // namespace kmm