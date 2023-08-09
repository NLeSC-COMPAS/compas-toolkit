#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "trajectories/spiral_kernels.cuh"

namespace compas {

struct SpiralTrajectory {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    CudaArray<cfloat> k_start;
    CudaArray<cfloat> delta_k;

    SpiralTrajectoryView view() const {
        return {
            .nreadouts = nreadouts,
            .samples_per_readout = samples_per_readout,
            .delta_t = delta_t,
            .k_start = k_start.view(),
            .delta_k = delta_k.view()};
    }
};

SpiralTrajectory make_spiral_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    host_view<cfloat> delta_k);

}  // namespace compas