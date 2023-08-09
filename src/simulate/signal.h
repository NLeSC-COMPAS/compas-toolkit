#pragma once

#include "core/context.h"
#include "parameters/tissue.h"
#include "trajectories/multi.h"

namespace compas {

void simulate_signal(
    const CudaContext& context,
    CudaArray<cfloat, 3> signal,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    Trajectory trajectory,
    CudaArray<float, 2> coil_sensitivities);

}