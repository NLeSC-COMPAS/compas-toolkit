#pragma once

#include "signal.h"

namespace compas {

__global__ void to_sample_point_components(
        float2 m,
        Trajectory trajectory,
        int readout,
        int voxel,
        TissueParameters p
) {
    auto R_2 = 1 / p.T_2[voxel];
    auto ns = trajectory.nsamples_per_readout
    auto delta_t = trajectory.delta_t[readout];
    auto k_start = trajectory.k_start[readout]
    auto delta_k = trajectory.Δk_adc[readout_idx];

    auto xyz = p.xyz[voxel];
    auto x = xyz.x;
    auto y = xyz.y

    m = rewind(m, R2, float(0.5) * ns * delta_t, p);
    m = prephaser(m, k_start.re, k_start.im, x, y);

    auto theta = delta_k.re * x + delta_k.im * y
    if (has_B0(p)) {
        theta += π * p.B0 * Δt * 2;
    }

    lnE₂eⁱᶿ = (-Δt * R_2 + im*theta);
    return (m, lnE₂eⁱᶿ);
}

__global__ void simulate_prepare_kernel(
    int nreadouts,
    int nvoxels,
    CudaView<float, 2> exponents,
    CudaView<float, 2> factors,
    CudaView<float2, 2> echos
) {
    int readout = blockIdx.x * blockDim.x + threadIdx.x;
    int voxel = blockIdx.y * blockDim.y + threadIdx.y;

    if (readout >= nreadouts || voxel >= nvoxels) {
        return;
    }

    auto m = echos(voxel, readout);
    auto p = parameters(voxel)
    m_s, exponent = to_sample_point_components(m, trajectory, readout, p)

    exponents(voxel, readout) = exponent;
    factors(voxel, readout) = m_s * complex(p.ρˣ, p.ρʸ);
}

}