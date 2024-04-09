#pragma once

#include "core/complex_type.hpp"

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