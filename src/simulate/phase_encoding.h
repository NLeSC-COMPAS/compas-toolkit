#pragma once

#include "core/complex_type.hpp"

namespace compas {

/**
 * TODO: document function
 */
Array<cfloat, 2> phase_encoding(
    const CudaContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const Trajectory& trajectory
    );

} // compas