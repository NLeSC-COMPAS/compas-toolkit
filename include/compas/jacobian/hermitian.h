#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/jacobian/product.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

/**
 * @brief Computes the product Jᴴv, where J is the Jacobian w.r.t. tissue parameters.
 *
 * @param ctx CompasContext for execution resources.
 * @param echos Magnetization at echo times [nreadouts, nvoxels].
 * @param delta_echos_T1 Partial derivatives of `echos` w.r.t. T1 [nreadouts, nvoxels].
 * @param delta_echos_T2 Partial derivatives of `echos` w.r.t. T2 [nreadouts, nvoxels].
 * @param parameters TissueParameters struct [nvoxels].
 * @param trajectory CartesianTrajectory object.
 * @param coil_sensitivities Coil sensitivities [ncoils, nvoxels].
 * @param vector Input vector `v` in k-space [ncoils, nreadouts, nsamples_per_readout].
 * @return Result `Jᴴv` [nfields, nvoxels]. Fields are derivatives w.r.t. [T1, T2, rho_real, rho_imag].
 */
Array<cfloat, 2> compute_jacobian_hermitian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector,
    JacobianComputeMethod kind = JacobianComputeMethod::Direct);
}  // namespace compas