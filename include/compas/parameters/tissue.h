#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/core/object.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

/**
 * @brief Stores tissue parameters (T1, T2, B1, B0, rho, position) for multiple voxels.
 *
 * Data is stored in a 2D array `data` [num_parameters, nvoxels].
 * Parameter order depends on creation (e.g., via `make_tissue_parameters`).
 */
struct TissueParameters: public Object {
    /// @brief 2D array [num_parameters, nvoxels] holding all tissue parameters.
    Array<float, 2> data;
    /// @brief Total number of voxels.
    int nvoxels;
    /// @brief Voxel processing chunk size.
    int chunk_size;
    /// @brief True if z-coordinates are included in `data`.
    bool has_z = true;
    /// @brief True if B0 off-resonance values are included in `data`.
    bool has_b0 = true;
    /// @brief True if B1 scaling values are included in `data`.
    bool has_b1 = true;

    TissueParameters(Array<float, 2> parameters, bool has_z, bool has_b0, bool has_b1) :
        data(parameters),
        nvoxels(kmm::checked_cast<int>(parameters.size(1))),
        chunk_size(kmm::checked_cast<int>(parameters.chunk_size(1))),
        has_z(has_z),
        has_b0(has_b0),
        has_b1(has_b1) {}
};

/**
 * @brief Creates TissueParameters from individual parameters.
 * @param ctx CompasContext for resource management.
 * @param num_voxels Total number of voxels.
 * @param chunk_size Desired processing chunk size.
 * @param T1 T1 relaxation times.
 * @param T2 T2 relaxation times.
 * @param B1 B1 transmit field scaling factors.
 * @param B0 B0 off-resonance field.
 * @param rho_x Real part of proton density/coil sensitivity.
 * @param rho_y Imaginary part of proton density/coil sensitivity.
 * @param x X-coordinates.
 * @param y Y-coordinates.
 * @param z Optional z-coordinates. If empty, `has_z` becomes false.
 * @return TissueParameters object.
 */
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