#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

/**
 * Apply phase encoding to the given magnetization.
 *
 * @param ctx Compas context.
 * @param echos The magnetization at echo times. Size: `[nreadouts, nvoxels]`.
 * @param parameters The tissue parameters.
 * @param trajectory The `Trajectory`.
 * @return The phase encoded magnetization at echo times.
 */
Array<cfloat, 2> phase_encoding(
    const CompasContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const CartesianTrajectory& trajectory);

}  // namespace compas