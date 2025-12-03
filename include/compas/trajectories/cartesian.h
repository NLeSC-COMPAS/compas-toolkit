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
    const CompasContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    HostView<cfloat> k_start,
    cfloat delta_k) {
    COMPAS_CHECK(k_start.size() == nreadouts);

    return CartesianTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        delta_k};
}

}  // namespace compas

KMM_DEFINE_STRUCT_ARGUMENT(
    compas::CartesianTrajectory,
    it.nreadouts,
    it.samples_per_readout,
    it.delta_t,
    it.k_start,
    it.delta_k)

KMM_DEFINE_STRUCT_VIEW(compas::CartesianTrajectory, compas::CartesianTrajectoryView)