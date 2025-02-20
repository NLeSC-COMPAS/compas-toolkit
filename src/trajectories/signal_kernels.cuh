#pragma once

#include "compas/core/view.h"
#include "compas/trajectories/cartesian_view.cuh"
#include "compas/trajectories/spiral_view.cuh"

namespace compas {
namespace kernels {

template<typename TrajectoryView>
__global__ void prepare_signal_factors(
    kmm::Range<index_t> voxels,
    kmm::Range<index_t> readouts,
    GPUSubviewMut<cfloat, 2> factors,
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    TrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + voxels.begin);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y + readouts.begin);

    if (voxel < voxels.end && readout < readouts.end) {
        auto m = echos[readout][voxel];

        auto p = parameters.get(voxel);
        auto ms = trajectory.to_sample_point_factor(readout, m, p);
        auto factor = ms * p.rho;

        factors[readout][voxel] = factor;
    }
}

__global__ void prepare_signal_cartesian(
    kmm::Range<index_t> voxels,
    int num_samples,
    GPUSubviewMut<cfloat, 2> exponents,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + voxels.begin);

    if (voxel < voxels.end) {
        auto p = parameters.get(voxel);
        auto exponent = trajectory.to_sample_point_exponent(p);

        for (int sample = 0; sample < num_samples; sample++) {
            exponents[sample][voxel] = exp(exponent * float(sample));
        }
    }
}

__global__ void prepare_signal_cartesian_with_coil(
    GPUViewMut<cfloat, 2> exponents,
    GPUView<cfloat> coil_sensitivities,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto num_samples = exponents.size(0);
    auto num_voxels = exponents.size(1);

    if (voxel < num_voxels) {
        auto p = parameters.get(voxel);
        auto exponent = trajectory.to_sample_point_exponent(p);
        auto coil = coil_sensitivities[voxel];

        for (int sample = 0; sample < num_samples; sample++) {
            exponents[sample][voxel] = coil * exp(exponent * float(sample));
        }
    }
}

__global__ void prepare_signal_spiral(
    GPUViewMut<cfloat, 2> exponents,
    TissueParametersView parameters,
    SpiralTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (readout < exponents.size(0) && voxel < exponents.size(1)) {
        auto p = parameters.get(voxel);
        auto exponent = trajectory.to_sample_point_exponent(readout, p);

        exponents[readout][voxel] = exponent;
    }
}

template<int threads_cooperative>
COMPAS_DEVICE void reduce_signal_cooperative(
    int sample_index,
    int readout_index,
    int coil_index,
    GPUSubviewMut<cfloat, 3> signal,
    int lane,
    cfloat value) {
    static_assert(is_power_of_two(threads_cooperative) && threads_cooperative <= 32);

#pragma unroll 6
    for (uint delta = threads_cooperative / 2; delta > 0; delta /= 2) {
        static constexpr uint mask = uint((1L << threads_cooperative) - 1);

        value.re += __shfl_down_sync(mask, value.re, delta, threads_cooperative);
        value.im += __shfl_down_sync(mask, value.im, delta, threads_cooperative);
    }

    // Only thread zero writes the results
    if (lane == 0) {
        if (signal.in_bounds({coil_index, readout_index, sample_index})) {
            signal[coil_index][readout_index][sample_index] = value;
        }
    }
}

template<
    int threads_per_block,
    int threads_cooperative,
    int sample_tiling_factor,
    int readout_tiling_factor,
    int coil_tiling_factor,
    int blocks_per_sm = 1>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void sum_signal_cartesian(
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,  // [num_coils num_readouts num_samples]
    GPUSubview<cfloat, 2> exponents,  // [num_samples num_voxels]
    GPUSubview<cfloat, 2> factors,  // [num_readouts num_voxels]
    GPUSubview<cfloat, 2> coil_sensitivities  // [num_coils num_voxels]
) {
    static_assert(threads_per_block % threads_cooperative == 0);

    auto num_coils = signal.size(0);
    auto num_readouts = signal.size(1);

    auto lane = index_t(threadIdx.x % threads_cooperative);
    auto sample_start = index_t((blockIdx.x * blockDim.x + threadIdx.x) / threads_cooperative)
        * sample_tiling_factor;
    auto readout_start = index_t(blockIdx.y * blockDim.y + threadIdx.y) * readout_tiling_factor;
    auto coil_start = index_t(blockIdx.z * blockDim.z + threadIdx.z) * coil_tiling_factor;

    // If the start indices are out of bounds, we exit immediately
    if (!signal.in_bounds({coil_start, readout_start, sample_start})) {
        return;
    }

    cfloat sums[coil_tiling_factor][readout_tiling_factor][sample_tiling_factor];
#pragma unroll
    for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
        for (int r = 0; r < readout_tiling_factor; r++) {
#pragma unroll
            for (int c = 0; c < coil_tiling_factor; c++) {
                sums[c][r][s] = 0;
            }
        }
    }

    for (index_t voxel = voxels.begin + lane; voxel < voxels.end; voxel += threads_cooperative) {
        cfloat local_coils[coil_tiling_factor] = {0};

#pragma unroll
        for (int c = 0; c < coil_tiling_factor; c++) {
            auto coil_index = coil_start + c;
            local_coils[c] =
                c == 0 || coil_index < num_coils ? coil_sensitivities[coil_index][voxel] : 0;
        }

        cfloat step = exponents[1][voxel];
        cfloat exponent = exponents[sample_start][voxel];

#pragma unroll
        for (int r = 0; r < readout_tiling_factor; r++) {
            int readout = readout_start + r;
            cfloat local_sample =
                r == 0 || readout < num_readouts ? factors[readout][voxel] * exponent : 0;

#pragma unroll
            for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
                for (int c = 0; c < coil_tiling_factor; c++) {
                    auto coil = local_coils[c];
                    auto sample = local_sample;
                    sums[c][r][s] += sample * coil;
                }

                local_sample *= step;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
        for (int r = 0; r < readout_tiling_factor; r++) {
#pragma unroll
            for (int c = 0; c < coil_tiling_factor; c++) {
                reduce_signal_cooperative<threads_cooperative>(
                    sample_start + s,
                    readout_start + r,
                    coil_start + c,
                    signal,
                    lane,
                    sums[c][r][s]);
            }
        }
    }
}

template<
    int threads_per_block,
    int threads_cooperative,
    int sample_tiling_factor,
    int coil_tiling_factor,
    int blocks_per_sm = 1>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void sum_signal_spiral(
    GPUViewMut<cfloat, 3> signal,  // [num_coils num_readouts num_samples]
    GPUView<cfloat, 2> exponents,  // [num_readouts num_voxels]
    GPUView<cfloat, 2> factors,  // [num_readouts num_voxels]
    GPUView<cfloat, 2> coil_sensitivities  // [num_coils num_voxels]
) {
    static_assert(threads_per_block % threads_cooperative == 0);

    auto num_voxels = coil_sensitivities.size(1);
    auto num_coils = coil_sensitivities.size(0);

    auto lane = index_t(threadIdx.x % threads_cooperative);
    auto sample_start =
        index_t((blockIdx.x * threads_per_block + threadIdx.x) / threads_cooperative)
        * sample_tiling_factor;
    auto readout = index_t(blockIdx.y);
    auto coil_start = index_t(blockIdx.z) * coil_tiling_factor;

    cfloat sums[coil_tiling_factor][sample_tiling_factor];
    for (int s = 0; s < sample_tiling_factor; s++) {
        for (int c = 0; c < coil_tiling_factor; c++) {
            sums[c][s] = 0;
        }
    }

    for (index_t voxel = lane; voxel < num_voxels; voxel += threads_cooperative) {
        cfloat local_coils[coil_tiling_factor] = {0};

#pragma unroll
        for (int c = 0; c < coil_tiling_factor; c++) {
            auto coil_index = coil_start + c;
            local_coils[c] = coil_index < num_coils ? coil_sensitivities[coil_index][voxel] : 0;
        }

        auto exponent = exponents[readout][voxel];
        auto factor = factors[readout][voxel];

        auto step = exp(exponent);
        auto base = exp(exponent * float(sample_start)) * factor;

#pragma unroll
        for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
            for (int c = 0; c < coil_tiling_factor; c++) {
                auto coil = local_coils[c];
                sums[c][s] += base * coil;
            }

            base *= step;
        }
    }

#pragma unroll
    for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
        for (int c = 0; c < coil_tiling_factor; c++) {
            reduce_signal_cooperative<threads_cooperative>(
                sample_start + s,
                readout,
                coil_start + c,
                signal,
                lane,
                sums[c][s]);
        }
    }
}
}  // namespace kernels
}  // namespace compas
