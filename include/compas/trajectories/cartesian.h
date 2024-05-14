#pragma once

#include "compas/core/assertion.h"
#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/trajectories/cartesian_view.cuh"
#include "compas/trajectories/trajectory.h"

namespace compas {

/**
 *  Describes a Cartesian gradient trajectory. This is represented by storing the starting position in `k-space` for
 *  each readout and the step (`delta_k`) per sample point. It is assumed that the step is the same for all readouts.
 *  Note that the cartesian trajectory is a special case of the spiral trajectory, having the same step size for each
 *  readout (meaning it can be represented using a single value instead of an array of values).
 */
struct CartesianTrajectory: public Trajectory {
    Array<cfloat> k_start;  // Size: nreadouts
    cfloat delta_k;  // Size: nreadouts

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

/**
 * Create a Cartesian trajectory object.
 *
 * @param context
 * @param nreadouts
 * @param samples_per_readout
 * @param delta_t
 * @param k_start
 * @param delta_k
 * @return
 */
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