#pragma once

#include "compas/trajectories/cartesian_view.cuh"
#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

struct SpiralTrajectoryView {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    cuda_view<cfloat> k_start;
    cuda_view<cfloat> delta_k;

    COMPAS_DEVICE
    cfloat to_sample_point_factor(index_t readout_idx, cfloat m, TissueVoxel p) const {
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
    cfloat to_sample_point_exponent(index_t readout_idx, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto delta_k0 = delta_k[readout_idx];
        auto x = p.x;
        auto y = p.y;

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k0.re * x + delta_k0.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return cfloat(-delta_t * R2, Theta);
    }
};

}  // namespace compas