#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "sequences/pssfp_kernels.cuh"

namespace compas {
namespace kernels {

namespace {
COMPAS_DEVICE
float calculate_E1(float delta_t, float T1) {
    return __expf(-delta_t * (1.0f / T1));
}

COMPAS_DEVICE
float calculate_E2(float delta_t, float T2) {
    return __expf(-delta_t * (1.0f / T2));
}
}  // namespace

template<typename Group>
COMPAS_DEVICE void simulate_pssfp_for_voxel(
    Group group,
    const pSSFPSequenceView& sequence,
    float z,
    cuda_strided_view_mut<cfloat> echos,
    TissueVoxel p) {
    auto T1 = p.T1;
    auto T2 = p.T2;

    auto gamma_dt_RF_ex = sequence.gamma_dt_RF;
    auto gamma_dt_GRz_ex = sequence.gamma_dt_GRz.ex;
    auto dt_ex = sequence.dt.ex;
    auto E1_ex = calculate_E1(dt_ex, T1);
    auto E2_ex = calculate_E2(dt_ex, T2);

    auto gamma_dt_GRz_pr = sequence.gamma_dt_GRz.pr;
    auto dt_pr = sequence.dt.pr;
    auto E1_pr = calculate_E1(dt_pr, T1);
    auto E2_pr = calculate_E2(dt_pr, T2);

    auto dt_inv = sequence.dt.inv;
    auto E1_inv = calculate_E1(dt_inv, T1);
    auto E2_inv = calculate_E2(dt_inv, T2);

    // Simulate excitation with flip angle theta using hard pulse approximation of the normalized RF-waveform γdtRF
    auto excite = [&](Isochromat m, cfloat theta, float z) {
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

template<int batch_size>
__global__ void simulate_pssfp(
    cuda_view_mut<cfloat, 2> echos,
    cuda_view<float> z,
    TissueParametersView parameters,
    pSSFPSequenceView sequence) {
    index_t voxel = index_t(blockDim.x * blockIdx.x + threadIdx.x) / batch_size;
    index_t nvoxels = parameters.nvoxels;

    auto group =
        cooperative_groups::tiled_partition<batch_size>(cooperative_groups::this_thread_block());

    if (voxel >= nvoxels) {
        return;
    }

    simulate_pssfp_for_voxel(
        group,
        sequence,
        z[group.thread_rank()],
        echos.drop_axis<1>(voxel),
        parameters.get(voxel));
}

}  // namespace kernels
}  // namespace compas