#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

enum struct JacobianComputeMethod { Naive, Direct, Gemm, GemmFast, GemmLow };

/**
 * Computes the product of the Jacobian matrix with the given vector.
 *
 * @param ctx The CUDA context.
 * @param echos The magnetization at echo time. Size: [nreadouts, nvoxels]
 * @param delta_echos_T1 The partial derivatives of `echo` with respect to T1.
 * @param delta_echos_T2 The partial derivatives of `echo` with respect to T2.
 * @param parameters The tissue parameters (T1, T2, etc.)
 * @param trajectory Cartesian k-space sampling trajectory.
 * @param coil_sensitivities The sensitivities of the receiver coils. Size: [ncoils, nvoxels]
 * @param vector The input vector. Size: [nfields, nvoxels]. There are 4 fields: T1, T2, and rho<sub>x</sub>/rho<sub>y</sub>.
 * @return The result of `Jv`. Size: [ncoils, nreadouts, nsamples_per_readout].
 */
Array<cfloat, 3> compute_jacobian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 2> vector,
    JacobianComputeMethod kind = JacobianComputeMethod::Direct);

}  // namespace compas