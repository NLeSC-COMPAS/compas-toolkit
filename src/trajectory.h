#pragma once

#include "context.h"
#include "utils/complex_type.h"

namespace compas {

struct CartesianTrajectory {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    CudaArray<cfloat> k_start;
    CudaArray<cfloat> delta_k;
};

CartesianTrajectory make_cartesian_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    CudaArray<cfloat> k_start,
    CudaArray<cfloat> delta_k);

}  // namespace compas