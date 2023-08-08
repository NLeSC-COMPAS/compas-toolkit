#include "core/view.h"
#include "trajectories/cartesian_kernels.cuh"

namespace compas {
namespace kernels {

__global__ void prepare_signal(
    cuda_view_mut<cfloat, 2> exponents,
    cuda_view_mut<cfloat, 2> factors,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (readout < exponents.size(0) && voxel < exponents.size(1)) {
        auto m = echos[readout][voxel];

        auto p = parameters.get(voxel);
        auto exponent = trajectory.to_sample_point_exponent(readout, p);
        auto ms = trajectory.to_sample_point_factor(readout, m, p);
        auto factor = ms * p.rho;

        exponents[readout][voxel] = exponent;
        factors[readout][voxel] = factor;
    }
}

template<
    int threads_per_block,
    int threads_cooperative,
    int sample_tiling_factor,
    int coil_tiling_factor,
    int blocks_per_sm = 1>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void sum_signal(
    cuda_view_mut<cfloat, 3> signal,  // [num_coils num_readouts num_samples]
    cuda_view<cfloat, 2> exponents,  // [num_readouts num_voxels]
    cuda_view<cfloat, 2> factors,  // [num_readouts num_voxels]
    cuda_view<float, 2> coil_sensitivities  // [num_coils num_voxels]
) {
    static_assert(is_power_of_two(threads_cooperative) && threads_cooperative <= 32);
    static_assert(threads_per_block % threads_cooperative == 0);

    auto num_samples = signal.size(2);
    auto num_voxels = coil_sensitivities.size(1);
    auto num_coils = coil_sensitivities.size(0);

    auto voxel_start = index_t(threadIdx.x % threads_cooperative);
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

    for (index_t voxel = voxel_start; voxel < num_voxels; voxel += threads_cooperative) {
        float local_coils[coil_tiling_factor] = {0};

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
            auto coil_index = coil_start + c;
            auto sample = sample_start + s;
            auto value = sums[c][s];

#pragma unroll 6
            for (uint delta = threads_cooperative / 2; delta > 0; delta /= 2) {
                static constexpr uint mask = uint((1L << threads_cooperative) - 1);

                value.re += __shfl_down_sync(mask, value.re, delta, threads_cooperative);
                value.im += __shfl_down_sync(mask, value.im, delta, threads_cooperative);
            }

            // Only thread zero writes the results
            if (threadIdx.x == 0) {
                if (sample < num_samples && coil_index < num_coils) {
                    signal[coil_index][readout][sample] = value;
                }
            }
        }
    }
}
}  // namespace kernels
}  // namespace compas
