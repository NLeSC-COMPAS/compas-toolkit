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
    View<cfloat> k_start,
    View<cfloat> delta_k) {
    COMPAS_CHECK(k_start.size() == nreadouts);
    COMPAS_CHECK(delta_k.size() == nreadouts);

    return SpiralTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        context.allocate(delta_k)};
}

}  // namespace compas

KMM_DEFINE_STRUCT_ARGUMENT(
    compas::SpiralTrajectory,
    it.nreadouts,
    it.samples_per_readout,
    it.delta_t,
    it.k_start,
    it.delta_k)

KMM_DEFINE_STRUCT_VIEW(compas::SpiralTrajectory, compas::SpiralTrajectoryView)