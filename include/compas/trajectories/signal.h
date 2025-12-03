#pragma once

#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/spiral_view.cuh"
#include "compas/trajectories/trajectory.h"

namespace compas {

/**
 * Compute method used for `magnetization_to_signal`. Some method yield better performance, but at the cost of less
 * accurate results.
 */
enum struct SimulateSignalMethod {
    /**
     * Naive simulation without optimization or approximations. This is the most reliable method, but is also slow.
     */
    Naive,

    /**
     * Direct simulation. This is the more performant method if there are many coils.
     */
    Direct,

    /**
     * Use highest precision matrix multiplication method. This ensure highest precision.
     */
    MatmulPedantic,

    /**
     * Use matrix multiplication method. This ensure high accuracy with reasonable performance.
     */
    Matmul,

    /**
     * Use fast matrix multiplication method. This ensure reasonable accuracy with decent performance.
     */
    MatmulFast,

    /**
     * Use low precision matrix multiplication method using bfloat16. This gives low accuracy with highest performance.
     */
    MatmulLow,
};

/**
 * Compute the MR signal given the magnetization at echo times in all voxels.
 *
 * @param context Compas Context.
 * @param echos Magnetization at echo times. Size: [nreadouts, nvoxels].
 * @param parameters The tissue parameters.
 * @param trajectory The trajectory.
 * @param coil_sensitivities The coil sensitivities. Size: [ncoils, nvoxels].
 * @param method Method used for calculating. See `SimulateSignalMethod`.
 * @return The MR signal. Size: [ncoils, nreadouts, nsamples_per_readout].
 */
Array<cfloat, 3> magnetization_to_signal(
    const CompasContext& context,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    const Trajectory& trajectory,
    Array<cfloat, 2> coil_sensitivities,
    SimulateSignalMethod method = SimulateSignalMethod::MatmulFast);

}  // namespace compas