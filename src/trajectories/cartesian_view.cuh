#pragma once

#include "core/complex_type.h"
#include "core/view.h"
#include "parameters/tissue_view.cuh"

namespace compas {

// given magnetization at echo time,
// undo T2 decay and B0 phase that happened between start readout and echo time
COMPAS_DEVICE
cfloat rewind(cfloat m, float R2, float delta_t, TissueVoxel p) {
    // m is magnetization at echo time
    // undo T2 decay and B0 phase that happened between start readout and echo time
    cfloat arg = cfloat(R2, -float(2 * M_PI) * p.B0);
    return m * exp(delta_t * arg);
}

// apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
COMPAS_DEVICE
cfloat prephaser(cfloat m, float k_x, float k_y, float x, float y) {
    return m * exp(cfloat(0, k_x * x + k_y * y));
}

struct CartesianTrajectoryView {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    cuda_view<cfloat> k_start;
    cfloat delta_k;

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
    cfloat to_sample_point_exponent(TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto x = p.x;
        auto y = p.y;

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k.re * x + delta_k.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return cfloat(-delta_t * R2, Theta);
    }
};

}  // namespace compas