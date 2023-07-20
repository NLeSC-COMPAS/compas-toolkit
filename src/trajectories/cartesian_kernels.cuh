#pragma once

#include "core/complex_type.h"
#include "core/view.h"
#include "parameters/tissue_kernels.cuh"

namespace compas {

struct SampleComponents {
    cfloat m;
    cfloat exponent;
};

// given magnetization at echo time,
// undo T2 decay and B0 phase that happened between start readout and echo time
COMPAS_DEVICE
cfloat rewind(cfloat m, float R2, float delta_t, TissueVoxel p) {
    // m is magnetization at echo time
    // undo T2 decay and B0 phase that happened between start readout and echo time
    cfloat arg = delta_t * cfloat(R2, -float(2 * M_PI) * p.B0);
    return m * exp(arg);
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
    cuda_view<cfloat> delta_k;

    COMPAS_DEVICE
    SampleComponents to_sample_point_components(
        index_t readout_idx,
        cfloat m,
        TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto ns = samples_per_readout;
        auto k0 = k_start[readout_idx];
        auto delta_k0 = delta_k[readout_idx];
        auto x = p.x;
        auto y = p.y;

        // go back in time to the start of the readout by undoing T₂ decay and B₀ rotation
        m = rewind(m, R2, float(0.5 * ns) * delta_t, p);

        // apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
        m = prephaser(m, k0.re, k0.im, x, y);

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k0.re * x + delta_k0.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        // lnE2eⁱᶿ
        auto exponent = cfloat(-delta_t * R2, Theta);

        return {m, exponent};
    }
};

}  // namespace compas