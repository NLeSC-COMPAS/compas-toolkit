#pragma once

#include "compas/core/backends.h"
#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {

/** Given magnetization at echo time, undo T2 decay and B0 phase that happened between start readout and echo time.
  */
COMPAS_DEVICE
cfloat rewind(cfloat m, float R2, float delta_t, TissueVoxel p) {
    // m is magnetization at echo time
    // undo T2 decay and B0 phase that happened between start readout and echo time
    cfloat arg = cfloat(R2, -float(2 * M_PI) * p.B0);
    return m * exp(delta_t * arg);
}

/**
 *
 * Apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian).
 *
 * @param m
 * @param k_x
 * @param k_y
 * @param x
 * @param y
 * @return
 */
COMPAS_DEVICE
cfloat prephaser(cfloat m, float k_x, float k_y, float x, float y) {
    return m * exp(cfloat(0, k_x * x /*+ k_y * y*/));
}

struct SpiralTrajectoryView {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    GPUView<cfloat> k_start;
    GPUView<cfloat> delta_k;

    COMPAS_DEVICE
    cfloat calculate_readout_magnetization(index_t readout_idx, cfloat m, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto ns = samples_per_readout;
        auto k0 = k_start[readout_idx];
        auto x = p.x;
        auto y = p.y;

        // go back in time to the start of the readout by undoing T₂ decay and B₀ rotation
        m = rewind(m, R2, float(0.5 * ns) * delta_t, p);

        // apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
        m = prephaser(m, k0.re, k0.im, x, y);

        return m;
    }

    COMPAS_DEVICE
    cfloat calculate_sample_phase_decay(index_t readout_idx, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto delta_k0 = delta_k[readout_idx];
        auto x = p.x;
        auto y = p.y;

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k0.re * x + delta_k0.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return exp(cfloat(-delta_t * R2, Theta));
    }
};

}  // namespace compas