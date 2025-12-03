#pragma once

#include "../operators/epg.cuh"
#include "compas/parameters/tissue_view.cuh"
#include "compas/sequences/fisp_view.h"

namespace compas {
namespace kernels {

template<int warp_size, int warps_per_block>
struct FISPSharedMemory {
    float2 echos[warps_per_block][warp_size];
    EPGExciteMatrix excite_matrix[warps_per_block][warp_size];
};

template<
    bool NumStepsIsWarpSize,
    bool SampleTransverse,
    int max_N,
    int warp_size,
    int warps_per_block>
COMPAS_DEVICE void simulate_fisp_for_voxel_repetition_steps_warp(
    int offset_step,
    int num_steps,
    EPGThreadBlockState<max_N, warp_size, warps_per_block>& omega,
    FISPSharedMemory<warp_size, warps_per_block>& smem,
    const FISPSequenceView& sequence,
    GPUView<cfloat> slice_profile,
    GPUSubviewMut<cfloat> echos,
    TissueVoxel p,
    float E1_TE,
    float E2_TE,
    cfloat r_TE,
    float E1_TR_minus_TE,
    float E2_TR_minus_TE,
    cfloat r_TR_minus_TE) {
    if (NumStepsIsWarpSize || threadIdx.x < num_steps) {
        smem.excite_matrix[threadIdx.y][threadIdx.x] = EPGExciteMatrix(
            slice_profile[offset_step + threadIdx.x] * sequence.RF_train[offset_step + threadIdx.x],
            p.B1);
    }
#if defined(COMPAS_USE_HIP)
    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<warp_size>(block);
#endif

#if defined(COMPAS_USE_CUDA)
    __syncwarp();
#elif defined(COMPAS_USE_HIP)
    warp.sync();
#endif

#pragma unroll warp_size
    for (index_t j = 0; j < warp_size; j++) {
        if (!NumStepsIsWarpSize && j >= num_steps) {
            break;
        }

        // mix states
        // omega.excite(slice_profile[offset + j] * sequence.RF_train[offset + j], p.B1);
        omega.excite(smem.excite_matrix[threadIdx.y][j]);

        // T2 decay F states, T1 decay Z states, B0 rotation until TE
        omega.rotate_decay(E1_TE, E2_TE, r_TE);
        omega.regrowth(E1_TE);

        // omega.sample_transverse(&echos[R * (offset + j)], 0);
        if constexpr (SampleTransverse) {
            cfloat result;

            if (omega.sample_transverse(0, &result)) {
                // echos[offset_step + j] += result;
                smem.echos[threadIdx.y][j] = float2 {result.re, result.im};
            }
        }

        // T2 decay F states, T1 decay Z states, B0 rotation until next RF excitation
        omega.rotate_decay(E1_TR_minus_TE, E2_TR_minus_TE, r_TR_minus_TE);
        omega.regrowth(E1_TR_minus_TE);
        // shift F states due to dephasing gradients
        omega.dephasing();
    }

#if defined(COMPAS_USE_CUDA)
    __syncwarp();
#elif defined(COMPAS_USE_HIP)
    warp.sync();
#endif

    if constexpr (SampleTransverse) {
        auto e = smem.echos[threadIdx.y][threadIdx.x];

        if (NumStepsIsWarpSize || threadIdx.x < num_steps) {
            echos[offset_step + threadIdx.x] += cfloat(e.x, e.y);
        }
    }
}

template<bool SampleTransverse, int max_N, int warp_size, int warps_per_block>
COMPAS_DEVICE void simulate_fisp_for_voxel_repetition(
    EPGThreadBlockState<max_N, warp_size, warps_per_block>& omega,
    FISPSharedMemory<warp_size, warps_per_block>& smem,
    const FISPSequenceView& sequence,
    GPUView<cfloat> slice_profile,
    GPUSubviewMut<cfloat> echos,
    TissueVoxel p) {
    KMM_ASSUME(threadIdx.x < warp_size);
    KMM_ASSUME(threadIdx.y < warps_per_block);

    auto off_resonance_rotation = [](float delta_t, float B0 = 0.0f) -> cfloat {
        if (B0 == 0.0f) {
            return 1;
        }

        auto theta = float(2.0f * M_PI) * delta_t * B0;
        return polar(1.0f, theta);
    };

    auto calculate_E = [](float delta_t, float T) -> float {
        return __expf(-delta_t * (1.0f / T));
    };

    auto T1 = p.T1;
    auto T2 = p.T2;
    auto TR = sequence.TR;
    auto TE = sequence.TE;
    auto TI = sequence.TI;
    auto TW = sequence.TW;

    auto E1_TE = calculate_E(TE, T1);
    auto E2_TE = calculate_E(TE, T2);

    auto E1_TI = calculate_E(TI, T1);
    auto E2_TI = calculate_E(TI, T2);

    auto E1_TR_minus_TE = calculate_E(TR - TE, T1);
    auto E2_TR_minus_TE = calculate_E(TR - TE, T2);

    auto E1_W = calculate_E(TW, T1);
    auto E2_W = calculate_E(TW, T2);

    auto r_TE = off_resonance_rotation(TE, p.B0);
    auto r_TR_minus_TE = off_resonance_rotation(TR - TE, p.B0);
    auto r_TW = off_resonance_rotation(TW, p.B0);

    // apply inversion pulse
    if (sequence.inversion_prepulse) {
        omega.invert();
        omega.spoil();
        omega.decay(E1_TI, E2_TI);
        omega.regrowth(E1_TI);
    }

    for (index_t offset = 0;; offset += warp_size) {
        int steps_remaining = int(sequence.RF_train.size()) - offset;

        if (steps_remaining >= warp_size) {
            simulate_fisp_for_voxel_repetition_steps_warp<true, SampleTransverse>(
                offset,
                warp_size,
                omega,
                smem,
                sequence,
                slice_profile,
                echos,
                p,
                E1_TE,
                E2_TE,
                r_TE,
                E1_TR_minus_TE,
                E2_TR_minus_TE,
                r_TR_minus_TE);
        } else {
            simulate_fisp_for_voxel_repetition_steps_warp<false, SampleTransverse>(
                offset,
                steps_remaining,
                omega,
                smem,
                sequence,
                slice_profile,
                echos,
                p,
                E1_TE,
                E2_TE,
                r_TE,
                E1_TR_minus_TE,
                E2_TR_minus_TE,
                r_TR_minus_TE);

            // break as we are done with the loop
            break;
        }
    }

    if (sequence.wait_spoiling) {
        omega.spoil();
    }

    omega.rotate_decay(E1_W, E2_W, r_TW);
    omega.regrowth(E1_W);
}

template<int max_N, int warp_size, int warps_per_block>
COMPAS_DEVICE void simulate_fisp_for_voxel(
    const FISPSequenceView& sequence,
    GPUView<cfloat> slice_profile,
    GPUSubviewMut<cfloat> echos,
    TissueVoxel p) {
    // Prepare shared memory
    __shared__ FISPSharedMemory<warp_size, warps_per_block> smem;

    // Prepare the thread block local state
    auto omega = EPGThreadBlockState<max_N, warp_size, warps_per_block>(sequence.max_state);
    omega.initialize();

    // Perform `sequence.repetitions-1` repetitions
    for (index_t repeat = sequence.repetitions; repeat > 1; repeat--) {
        simulate_fisp_for_voxel_repetition<false>(omega, smem, sequence, slice_profile, echos, p);
    }

    // sample F₊[0] at last repetition
    simulate_fisp_for_voxel_repetition<true>(omega, smem, sequence, slice_profile, echos, p);
}

template<int max_N, int warp_size, int warps_per_block>
__launch_bounds__(warp_size* warps_per_block, 4) __global__ void simulate_fisp(
    kmm::Range<index_t> range,
    GPUSubviewMut<cfloat, 2> transposed_echos,
    GPUView<cfloat> slice_profile,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    index_t warp_id = index_t(blockDim.y * blockIdx.y + threadIdx.y);
    index_t voxel = range.begin + warp_id;

    if (voxel >= range.end) {
        return;
    }

    simulate_fisp_for_voxel<max_N, warp_size, warps_per_block>(
        sequence,
        slice_profile,
        transposed_echos.drop_axis(voxel),
        parameters.get(voxel));
}

__global__ void expand_undersampled_echos(
    kmm::Range<index_t> voxels,
    kmm::Range<index_t> readouts,
    int undersampling_factor,
    GPUSubviewMut<cfloat, 2> echos,
    GPUSubview<cfloat, 2> undersampled_transposed_echos) {
    index_t voxel_id = index_t(blockDim.x * blockIdx.x + threadIdx.x) + voxels.begin;
    index_t readout_id = index_t(blockDim.y * blockIdx.y + threadIdx.y) + readouts.begin;

    if (voxel_id < voxels.end && readout_id < readouts.end) {
        echos[readout_id][voxel_id] =
            undersampled_transposed_echos[voxel_id][readout_id / undersampling_factor];
    }
}

}  // namespace kernels
}  // namespace compas
