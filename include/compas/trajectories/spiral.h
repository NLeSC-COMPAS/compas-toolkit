#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/trajectories/spiral_view.cuh"
#include "compas/trajectories/trajectory.h"

namespace compas {

/**
 *  Describes a Spiral gradient trajectory. This is represented by storing the starting position in `k-space` for
 *  each readout and the step (`delta_k`) per sample point. The step can be different for each readout.
 */
struct SpiralTrajectory: public Trajectory {
    Array<cfloat> k_start;  // Size: nreadouts
    Array<cfloat> delta_k;  // Size: nreadouts

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
    const CompasContext& context,
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
struct Argument<compas::SpiralTrajectory> {
    using type = compas::SpiralTrajectoryView;

    static Argument pack(TaskBuilder& builder, const compas::SpiralTrajectory& t) {
        return {
            {//
             .nreadouts = t.nreadouts,
             .samples_per_readout = t.samples_per_readout,
             .delta_t = t.delta_t,
             .k_start = {},
             .delta_k = {}},
            pack_argument(builder, t.k_start),
            pack_argument(builder, t.delta_k)};
    }

    template<ExecutionSpace space>
    type unpack(TaskContext& context) {
        view.k_start = unpack_argument<space>(context, k_start);
        view.delta_k = unpack_argument<space>(context, delta_k);
        return view;
    }

    compas::SpiralTrajectoryView view;
    packed_argument_t<Array<compas::cfloat>> k_start;
    packed_argument_t<Array<compas::cfloat>> delta_k;
};
}  // namespace kmm