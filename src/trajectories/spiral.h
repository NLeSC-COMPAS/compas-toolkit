#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "trajectories/spiral_view.cuh"
#include "trajectories/trajectory.h"

namespace compas {

struct SpiralTrajectory: public Trajectory {
    CudaArray<cfloat> k_start;
    CudaArray<cfloat> delta_k;

    SpiralTrajectory(
        int nreadouts,
        int samples_per_readout,
        float delta_t,
        CudaArray<cfloat> k_start,
        CudaArray<cfloat> delta_k) :
        Trajectory(nreadouts, samples_per_readout, delta_t),
        k_start(k_start),
        delta_k(delta_k) {}
};

inline SpiralTrajectory make_spiral_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    host_view<cfloat> delta_k) {
    COMPAS_ASSERT(k_start.size() == nreadouts);
    COMPAS_ASSERT(delta_k.size() == nreadouts);

    return SpiralTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        context.allocate(delta_k)};
}

}  // namespace compas

namespace kmm {
template<>
struct TaskArgument<ExecutionSpace::Cuda, compas::SpiralTrajectory> {
    using type = compas::SpiralTrajectoryView;

    static TaskArgument
    pack(RuntimeImpl& rt, TaskRequirements& reqs, const compas::SpiralTrajectory& t) {
        return {
            {//
             .nreadouts = t.nreadouts,
             .samples_per_readout = t.samples_per_readout,
             .delta_t = t.delta_t,
             .k_start = {},
             .delta_k = {}},
            pack_argument<ExecutionSpace::Cuda>(rt, reqs, t.k_start),
            pack_argument<ExecutionSpace::Cuda>(rt, reqs, t.delta_k)};
    }

    type unpack(TaskContext& context) {
        view.k_start =
            unpack_argument<ExecutionSpace::Cuda, Array<compas::cfloat>>(context, k_start);
        view.delta_k =
            unpack_argument<ExecutionSpace::Cuda, Array<compas::cfloat>>(context, delta_k);
        return view;
    }

    compas::SpiralTrajectoryView view;
    PackedArray<const compas::cfloat> k_start;
    PackedArray<const compas::cfloat> delta_k;
};
}  // namespace kmm