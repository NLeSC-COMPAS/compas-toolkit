#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

/**
 * Object to represent a view of a Cartesian trajectory.
 */
struct CartesianTrajectoryView {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    GPUView<cfloat> k_start;
    cfloat delta_k;

    /**
     * Apply prephaser.
     *
     * @param readout_idx
     * @param m
     * @param p
     * @return
     */
    COMPAS_DEVICE
    cfloat calculate_readout_magnetization(index_t readout_idx, cfloat m, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto ns = samples_per_readout;
        auto x = p.x;
        auto y = p.y;

        // undo T2 decay and B0 phase that happened between start readout and echo time
        cfloat arg = cfloat(-R2, float(2 * M_PI) * p.B0);
        cfloat result = float(-0.5f * ns) * delta_t * arg;

        // apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
        auto k0 = float(-0.5f * ns) * delta_k;  //k_start[readout_idx];
        result += cfloat(0, k0.re * x /*+ k0.im * y*/);

        return m * exp(result);
    }

    /**
     * Apply readout gradient, T₂ decay and B₀ rotation.
     *
     * @param p
     * @return
     */
    COMPAS_DEVICE
    cfloat calculate_sample_phase_decay(int sample_index, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto x = p.x;
        auto y = p.y;

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k.re * x + delta_k.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return exp(cfloat(-delta_t * R2, Theta) * float(sample_index));
    }
};

}  // namespace compas