#pragma once

#include "core/assertion.h"
#include "core/complex_type.h"
#include "core/context.h"
#include "trajectories/cartesian_view.cuh"
#include "trajectories/trajectory.h"

namespace compas {

struct CartesianTrajectory: public Trajectory {
    Array<cfloat> k_start;
    cfloat delta_k;

    CartesianTrajectory(
        int nreadouts,
        int samples_per_readout,
        float delta_t,
        Array<cfloat> k_start,
        cfloat delta_k) :
        Trajectory(nreadouts, samples_per_readout, delta_t),
        k_start(k_start),
        delta_k(delta_k) {}
};

inline CartesianTrajectory make_cartesian_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    cfloat delta_k) {
    COMPAS_ASSERT(k_start.size() == nreadouts);

    return CartesianTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        delta_k};
}

}  // namespace compas

namespace kmm {
template<>
struct TaskArgument<ExecutionSpace::Cuda, compas::CartesianTrajectory> {
    using type = compas::CartesianTrajectoryView;

    static TaskArgument pack(TaskBuilder& builder, const compas::CartesianTrajectory& t) {
        return {
            {//
             .nreadouts = t.nreadouts,
             .samples_per_readout = t.samples_per_readout,
             .delta_t = t.delta_t,
             .k_start = {},
             .delta_k = t.delta_k},
            pack_argument<ExecutionSpace::Cuda>(builder, t.k_start)};
    }

    type unpack(TaskContext& context) {
        view.k_start = unpack_argument<ExecutionSpace::Cuda>(context, k_start);
        return view;
    }

    compas::CartesianTrajectoryView view;
    PackedArray<const compas::cfloat> k_start;
};
}  // namespace kmm