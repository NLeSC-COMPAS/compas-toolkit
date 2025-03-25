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

KMM_DEFINE_STRUCT_ARGUMENT(compas::TissueParameters, it.data)
KMM_DEFINE_STRUCT_VIEW(compas::TissueParameters, compas::TissueParametersView)

using TissueParametersSlice = kmm::Read<const compas::TissueParameters, kmm::Axis>;
KMM_DEFINE_STRUCT_ARGUMENT(TissueParametersSlice, it.argument.data[kmm::All()][it.access_mapper])
KMM_DEFINE_STRUCT_VIEW(TissueParametersSlice, compas::TissueParametersView)