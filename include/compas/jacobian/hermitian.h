#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

/**
 * Computes the product of the Hermitian transpose of the Jacobian matrix with the given vector.
 *
 * @param ctx
 * @param echos The magnetization at echo time. Size: [nreadouts, nvoxels]
 * @param delta_echos_T1 The partial derivatives of `echo` with respect to T1.
 * @param delta_echos_T2 The partial derivatives of `echo` with respect to T2.
 * @param parameters The tissue parameters (T1, T2, etc.)
 * @param trajectory Cartesian k-space sampling trajectory.
 * @param coil_sensitivities The sensitivities of the receiver coils. Size: [ncoils, nvoxels]
 * @param vector The input vector. Size: [ncoils, nreadouts, nsamples_per_readout]
 * @return The result of `Jᴴv`. Size: [nfields, nvoxels]. There are 4 fields: T1, T2, and rho<sub>x</sub>/rho<sub>y</sub>.
 */
Array<cfloat, 2> compute_jacobian_hermitian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector);
}  // namespace compas