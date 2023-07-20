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
        auto [ms, exponent] =
            trajectory.to_sample_point_components(readout, m, p);
        auto factor = ms * cfloat(p.rho.re, p.rho.im);

        exponents[readout][voxel] = exponent;
        factors[readout][voxel] = factor;
    }
}

__global__ void sum_signal(
    cuda_view_mut<cfloat, 2> signal,
    cuda_view<cfloat, 2> exponents,
    cuda_view<cfloat, 2> factors,
    cuda_view<float> coil_sensitivities) {
    auto num_readouts = signal.size(0);
    auto num_samples = signal.size(1);
    auto num_voxels = coil_sensitivities.size(0);

    auto sample = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto readout = index_t(blockIdx.y);

    if (readout < num_readouts && sample < num_samples) {
        cfloat sum = 0;

        for (index_t voxel = 0; voxel < num_voxels; voxel++) {
            auto exponent = exponents[readout][voxel];
            auto factor = factors[readout][voxel];
            auto coil = coil_sensitivities[voxel];

            sum += exp(exponent * float(sample)) * (factor * coil);
        }

        signal[readout][sample] = sum;
    }
}
}  // namespace kernels
}  // namespace compas
