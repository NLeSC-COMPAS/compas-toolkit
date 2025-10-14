#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/utils.h"
#include "compas/core/view.h"
#include "compas/trajectories/cartesian_view.cuh"
#include "compas/trajectories/spiral_view.cuh"

namespace compas {
namespace kernels {

template<typename TrajectoryView>
__global__ void prepare_readout_echos(
    kmm::Range<index_t> voxels,
    kmm::Range<index_t> readouts,
    GPUSubviewMut<cfloat, 2> readout_echos,
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    TrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x) + voxels.begin;
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y) + readouts.begin;

    if (voxel < voxels.end && readout < readouts.end) {
        auto m = echos[readout][voxel];
        auto p = parameters.get(voxel);
        auto ms = trajectory.calculate_readout_magnetization(readout, m, p);

        readout_echos[readout][voxel] = ms * p.rho;
    }
}

template<typename T>
__global__ void prepare_readout_echos_planar(
    kmm::Range<index_t> voxels,
    kmm::Range<index_t> readouts,
    GPUSubviewMut<T, 3> readout_echos,
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x) + voxels.begin;
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y) + readouts.begin;

    if (voxel < voxels.end && readout < readouts.end) {
        auto m = echos[readout][voxel];
        auto p = parameters.get(voxel);
        auto ms = trajectory.calculate_readout_magnetization(readout, m, p);

        readout_echos[0][readout][voxel] = kernel_float::cast<T>(real(ms * p.rho));
        readout_echos[1][readout][voxel] = kernel_float::cast<T>(imag(ms * p.rho));
    }
}

__global__ void prepare_sample_decay_cartesian(
    kmm::Range<index_t> voxels,
    int num_samples,
    GPUSubviewMut<cfloat, 2> sample_decay,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x) + voxels.begin;

    if (voxel < voxels.end) {
        auto p = parameters.get(voxel);

        for (int sample = 0; sample < num_samples; sample++) {
            auto decay = trajectory.calculate_sample_phase_decay(sample, p);
            sample_decay[sample][voxel] = decay;
        }
    }
}

template<typename T = float>
__global__ void prepare_sample_decay_cartesian_with_coil(
    kmm::Range<int> voxels,
    int num_samples,
    GPUViewMut<T, 3> sample_decay,  // planar complex
    GPUView<cfloat> coil_sensitivities,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x) + voxels.begin;

    if (voxel < voxels.end) {
        auto p = parameters.get(voxel);
        auto coil = coil_sensitivities[voxel];

        for (int sample = 0; sample < num_samples; sample++) {
            auto decay = trajectory.calculate_sample_phase_decay(sample, p);
            sample_decay[0][sample][voxel] = kernel_float::cast<T>(real(coil * decay));
            sample_decay[1][sample][voxel] = kernel_float::cast<T>(imag(coil * decay));
        }
    }
}

__global__ void prepare_sample_decay_spiral(
    GPUViewMut<cfloat, 2> sample_decay,
    TissueParametersView parameters,
    SpiralTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (readout < sample_decay.size(0) && voxel < sample_decay.size(1)) {
        auto p = parameters.get(voxel);
        auto decay = trajectory.calculate_sample_phase_decay(readout, p);
        sample_decay[readout][voxel] = decay;
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
    for (unsigned int delta = threads_cooperative / 2; delta > 0; delta /= 2) {
#if defined(COMPAS_IS_CUDA)
        static constexpr unsigned int mask = (1L << threads_cooperative) - 1;
#elif defined(COMPAS_IS_HIP)
        static constexpr long long unsigned int mask = (1L << threads_cooperative) - 1;
#endif

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

__global__ void sum_signal_cartesian_naive(
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,  // [num_coils num_readouts num_samples]
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    GPUSubview<cfloat, 2> coil_sensitivities  // [num_coils num_voxels]
) {
    auto sample_index = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout_index = index_t(blockIdx.y * blockDim.y + threadIdx.y);
    auto coil_index = index_t(blockIdx.z * blockDim.z + threadIdx.z);

    // If the start indices are out of bounds, we exit immediately
    if (!signal.in_bounds(coil_index, readout_index, sample_index)) {
        return;
    }

    cfloat sum = 0;

    for (index_t voxel = voxels.begin; voxel < voxels.end; voxel++) {
        auto p = parameters.get(voxel);
        auto m = echos[readout_index][voxel];
        auto ms = trajectory.calculate_readout_magnetization(readout_index, m, p);
        auto s = trajectory.calculate_sample_phase_decay(sample_index, p);
        auto c = coil_sensitivities[coil_index][voxel];

        sum += (ms * p.rho) * s * c;
    }

    signal[coil_index][readout_index][sample_index] = sum;
}

template<
    int threads_per_block,
    int threads_cooperative,
    int sample_tiling_factor,
    int readout_tiling_factor,
    int coil_tiling_factor,
    int blocks_per_sm = 1>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void sum_signal_cartesian(
    int voxel_begin,
    int num_voxels,
    int num_coils,
    int num_readouts,
    int num_samples,
    cfloat* signal_ptr,  // [num_coils num_readouts num_samples]
    const cfloat* sample_decay_ptr,  // [num_samples num_voxels]
    const cfloat* readout_echos_ptr,  // [num_readouts num_voxels]
    const cfloat* coil_sensitivities_ptr  // [num_coils num_voxels]
) {
    static_assert(threads_per_block % threads_cooperative == 0);

    GPUViewMut<cfloat, 3> signal = {signal_ptr, {{num_coils, num_readouts, num_samples}}};
    GPUSubview<cfloat, 2> sample_decay = {sample_decay_ptr, {{num_samples, num_voxels}}};
    GPUSubview<cfloat, 2> readout_echos = {readout_echos_ptr, {{num_readouts, num_voxels}}};
    GPUSubview<cfloat, 2> coil_sensitivities = {coil_sensitivities_ptr, {{num_coils, num_voxels}}};

    auto lane = index_t(threadIdx.x % threads_cooperative);
    auto sample_start = index_t((blockIdx.x * blockDim.x + threadIdx.x) / threads_cooperative)
        * sample_tiling_factor;
    auto readout_start = index_t(blockIdx.y * blockDim.y + threadIdx.y) * readout_tiling_factor;
    auto coil_start = index_t(blockIdx.z * blockDim.z + threadIdx.z) * coil_tiling_factor;

    // If the start indices are out of bounds, we exit immediately
    if (!signal.in_bounds(coil_start, readout_start, sample_start)) {
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

    for (index_t v = lane; v < num_voxels; v += threads_cooperative) {
        cfloat local_readouts[readout_tiling_factor];
        cfloat local_samples[sample_tiling_factor];
        cfloat local_coils[coil_tiling_factor];

        int voxel = voxel_begin + v;

#pragma unroll
        for (int r = 0; r < readout_tiling_factor; r++) {
            int readout = readout_start + r;
            local_readouts[r] =
                r == 0 || readout < num_readouts ? readout_echos[readout][voxel] : 0;
        }

#pragma unroll
        for (int s = 0; s < sample_tiling_factor; s++) {
            int sample = sample_start + s;
            local_samples[s] = s == 0 || sample < num_samples ? sample_decay[sample][voxel] : 0;
        }

#pragma unroll
        for (int c = 0; c < coil_tiling_factor; c++) {
            auto coil_index = coil_start + c;
            local_coils[c] =
                c == 0 || coil_index < num_coils ? coil_sensitivities[coil_index][voxel] : 0;
        }

#pragma unroll
        for (int r = 0; r < readout_tiling_factor; r++) {
#pragma unroll
            for (int s = 0; s < sample_tiling_factor; s++) {
#pragma unroll
                for (int c = 0; c < coil_tiling_factor; c++) {
                    sums[c][r][s] += (local_readouts[r] * local_samples[s]) * local_coils[c];
                }
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
    GPUView<cfloat, 2> sample_decay,  // [num_readouts num_voxels]
    GPUView<cfloat, 2> readout_echos,  // [num_readouts num_voxels]
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

        auto exponent = sample_decay[readout][voxel];
        auto factor = readout_echos[readout][voxel];

        auto step = exponent;
        auto base = pow(exponent, float(sample_start)) * factor;

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
