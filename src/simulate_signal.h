#pragma once

#include "context.h"
#include "tissueparameters.h"
#include "trajectory.h"

namespace compas {

void simulate_signal(
    const CudaContext& context,
    CudaArray<cfloat, 3> signal,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    CudaArray<cfloat, 2> coil_sensitivities);

}