#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

/**
 * Compute the phase encoding of a signal.
 */
Array<cfloat, 2> phase_encoding(
    const CudaContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const CartesianTrajectory& trajectory);

}  // namespace compas