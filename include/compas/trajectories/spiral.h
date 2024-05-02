#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/trajectories/spiral_view.cuh"
#include "compas/trajectories/trajectory.h"

namespace compas {

/**
 * Object representing a Spiral trajectory.
 */
struct SpiralTrajectory: public Trajectory {
    Array<cfloat> k_start;
    Array<cfloat> delta_k;

    SpiralTrajectory(
        int nreadouts,
        int samples_per_readout,
        float delta_t,
        Array<cfloat> k_start,
        Array<cfloat> delta_k) :
        Trajectory(nreadouts, samples_per_readout, delta_t),
        k_start(k_start),
        delta_k(delta_k) {}
};

/**
 *
 * Create a Spiral trajectory object.
 *
 * @param context
 * @param nreadouts
 * @param samples_per_readout
 * @param delta_t
 * @param k_start
 * @param delta_k
 * @return
 */
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

    static TaskArgument pack(TaskBuilder& builder, const compas::SpiralTrajectory& t) {
        return {
            {//
             .nreadouts = t.nreadouts,
             .samples_per_readout = t.samples_per_readout,
             .delta_t = t.delta_t,
             .k_start = {},
             .delta_k = {}},
            pack_argument<ExecutionSpace::Cuda>(builder, t.k_start),
            pack_argument<ExecutionSpace::Cuda>(builder, t.delta_k)};
    }

    type unpack(TaskContext& context) {
        view.k_start = unpack_argument<ExecutionSpace::Cuda>(context, k_start);
        view.delta_k = unpack_argument<ExecutionSpace::Cuda>(context, delta_k);
        return view;
    }

    compas::SpiralTrajectoryView view;
    PackedArray<const compas::cfloat> k_start;
    PackedArray<const compas::cfloat> delta_k;
};
}  // namespace kmm