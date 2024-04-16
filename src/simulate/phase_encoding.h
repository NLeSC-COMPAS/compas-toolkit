#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "parameters/tissue.h"
#include "trajectories/cartesian.h"

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