#pragma once

#if COMPAS_IS_CUDA
    #include <cooperative_groups/reduce.h>
#endif

#include "../operators/isochromat.cuh"
#include "compas/core/complex_type.h"
#include "compas/sequences/pssfp_view.h"

namespace compas {
namespace kernels {

template<int warp_size>
COMPAS_DEVICE void simulate_pssfp_for_voxel(
    const pSSFPSequenceView& sequence,
    float z,
    GPUSubviewStridedMut<cfloat> echos,
    TissueVoxel p) {
    auto group =
        cooperative_groups::tiled_partition<warp_size>(cooperative_groups::this_thread_block());

    auto calculate_E = [](float delta_t, float T) -> float {
        return __expf(-delta_t * (1.0f / T));
    };

    auto T1 = p.T1;
    auto T2 = p.T2;

    auto gamma_dt_RF_ex = sequence.gamma_dt_RF;
    auto gamma_dt_GRz_ex = sequence.gamma_dt_GRz.ex;
    auto dt_ex = sequence.dt.ex;
    auto E1_ex = calculate_E(dt_ex, T1);
    auto E2_ex = calculate_E(dt_ex, T2);

    auto gamma_dt_GRz_pr = sequence.gamma_dt_GRz.pr;
    auto dt_pr = sequence.dt.pr;
    auto E1_pr = calculate_E(dt_pr, T1);
    auto E2_pr = calculate_E(dt_pr, T2);

    auto dt_inv = sequence.dt.inv;
    auto E1_inv = calculate_E(dt_inv, T1);
    auto E2_inv = calculate_E(dt_inv, T2);

    // Simulate excitation with flip angle theta using hard pulse approximation of the normalized RF-waveform γdtRF
    auto excite = [&](Isochromat m, cfloat theta, float z) {
#pragma unroll 8
        for (index_t i = 0; i < gamma_dt_RF_ex.size(); i++) {
            auto zap = theta * gamma_dt_RF_ex[i];
            m = m.rotate(zap, gamma_dt_GRz_ex, z, dt_ex, p);
            m = m.decay(E1_ex, E2_ex);
            m = m.regrowth(E1_ex);
        }
        return m;
    };

    // Slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth
    auto precess = [&](Isochromat m, float z) {
        m = m.rotate(gamma_dt_GRz_pr, z, dt_pr, p);
        m = m.decay(E1_pr, E2_pr);
        m = m.regrowth(E1_pr);
        return m;
    };

    // reset spin to initial conditions
    Isochromat m;

    // apply inversion pulse
    m = m.invert(p);
    m = m.decay(E1_inv, E2_inv);
    m = m.regrowth(E1_inv);

    // apply "alpha over two" pulse
    auto theta0 = -sequence.RF_train[0] / float(2);
    m = excite(m, theta0, z);

    // slice select re- & prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
    m = m.rotate(float(2) * gamma_dt_GRz_pr, z, dt_pr, p);
    m = m.decay(E1_pr, E2_pr);
    m = m.regrowth(E1_pr);

    // simulate pSSFP sequence with varying flipangles
    for (index_t TR = 0; TR < sequence.RF_train.size(); TR++) {
        auto theta = sequence.RF_train[TR];

        // simulate RF pulse and slice-selection gradient
        m = excite(m, theta, z);

        // slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until TE
        m = precess(m, z);

        // sample magnetization at echo time (sum over slice direction)
        cfloat sum = {m.x, m.y};

#pragma unroll 32
        for (uint delta = group.size() / 2; delta > 0; delta /= 2) {
            sum.re += group.shfl_down(sum.re, delta);
            sum.im += group.shfl_down(sum.im, delta);
        }

        if (group.thread_rank() == 0) {
            echos[TR] += sum;
        }

        // slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
        m = precess(m, z);
    }
}

template<int warp_size>
__global__ void simulate_pssfp(
    index_t nvoxels,
    GPUSubviewMut<cfloat, 2> echos,
    GPUView<float> z,
    TissueParametersView parameters,
    pSSFPSequenceView sequence) {
    index_t lane = threadIdx.x % warp_size;
    index_t voxel = index_t(blockDim.x * blockIdx.x + threadIdx.x) / warp_size;

    if (voxel >= nvoxels) {
        return;
    }

    simulate_pssfp_for_voxel<warp_size>(
        sequence,
        z[lane],
        echos.drop_axis<1>(voxel),
        parameters.get(voxel));
}
}  // namespace kernels
}  // namespace compas
