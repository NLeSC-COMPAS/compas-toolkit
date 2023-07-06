#pragma once

#include "core/all.h"
#include "tissueparameters.h"

namespace compas {

struct Readout {
    int nsamples;
    float delta_t;
    cfloat k_start;
    cfloat delta_k;
};

struct SpokesTrajectory {
    int nsamples_per_readout;
    float delta_t;
    CudaView<const cfloat> k_start;
    CudaView<const cfloat> delta_k;

    COMPAS_DEVICE Readout get_readout(int readout_idx) const {
        return Readout {
            .nsamples = nsamples_per_readout,
            .delta_t = delta_t,
            .k_start = k_start[readout_idx],
            .delta_k = delta_k[readout_idx]};
    }
};

COMPAS_HOST_DEVICE
cfloat rewind(cfloat m, float R_2, float delta_t, Voxel p) {
    cfloat arg = delta_t * R_2;
    arg -= cfloat(0, -delta_t * 2.0f * float(M_PI) * p.B_0);
    return m * exp(arg);
}

COMPAS_HOST_DEVICE
cfloat prephaser(cfloat m, float k_x, float k_y, float x, float y) {
    return m * exp(cfloat(0, k_x * x + k_y * y));
}

COMPAS_HOST_DEVICE
pair<cfloat, cfloat>
to_sample_point_components(cfloat m, Readout readout, Voxel p) {
    auto R_2 = 1 / p.T_2;
    auto ns = readout.nsamples;
    auto delta_t = readout.delta_t;
    auto k_0 = readout.k_start;
    auto delta_k = readout.delta_k;
    auto x = p.x;
    auto y = p.y;

    m = rewind(m, R_2, 0.5f * ns * delta_t, p);
    m = prephaser(m, k_0.re, k_0.im, x, y);

    auto theta = delta_k.re * x + delta_k.im * y;
    theta += float(2 * M_PI) * p.B_0 * delta_t;

    auto lnE = cfloat(-delta_t * R_2, theta);

    return {m, lnE};
}

}  // namespace compas