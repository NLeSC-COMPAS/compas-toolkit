#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "trajectories/cartesian_view.cuh"

namespace compas {

struct CartesianTrajectory {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    CudaArray<cfloat> k_start;
    cfloat delta_k;

    CartesianTrajectoryView view() const {
        return {
            .nreadouts = nreadouts,
            .samples_per_readout = samples_per_readout,
            .delta_t = delta_t,
            .k_start = k_start.view(),
            .delta_k = delta_k};
    }
};

CartesianTrajectory make_cartesian_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    cfloat delta_k);

}  // namespace compas