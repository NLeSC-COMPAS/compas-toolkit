#pragma once

#include "core/context.h"
#include "core/complex_type.h"
#include "parameters/tissue.h"
#include "trajectories/trajectory.h"

namespace compas {

/**
 * Compute the phase encoding of a signal.
 */
Array<cfloat, 2> phase_encoding(
    const CudaContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const Trajectory& trajectory
    );

} // compas