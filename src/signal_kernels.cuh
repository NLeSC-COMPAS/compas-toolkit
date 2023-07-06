#include "core/all.h"
#include "trajectory.h"

namespace compas {

__global__ void simulate_signal_prepare_kernel(
    CudaView<cfloat, 2> exponents,
    CudaView<cfloat, 2> factors,
    CudaView<const cfloat, 2> echos,
    TissueParameters parameters,
    SpokesTrajectory trajectory) {
    int readout = blockIdx.x * blockDim.x + threadIdx.x;
    int voxel = blockIdx.y * blockDim.y + threadIdx.y;

    if (readout < echos.size(0) && voxel < echos.size(1)) {
        auto m = echos[readout][voxel];

        auto p = parameters.get_voxel(voxel);
        auto [m_s, exponent] =
            to_sample_point_components(m, trajectory.get_readout(readout), p);
        auto factor = m_s * p.rho;

        exponents[readout][voxel] = exponent;
        factors[readout][voxel] = factor;
    }
}

__global__ void simulate_signal_sum_kernel(
    CudaView<cfloat, 3> signal,
    CudaView<const cfloat, 2> exponents,
    CudaView<const cfloat, 2> factors,
    CudaView<const float, 2> coil_sensitivities) {
    int readout = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y * blockDim.y + threadIdx.y;
    int coil = blockIdx.z * blockDim.z + threadIdx.z;
    bool in_bounds = readout < signal.size(0) && sample < signal.size(1)
        && coil < signal.size(2);

    if (in_bounds) {
        int nv = exponents.size(1);
        cfloat S = 0;

        for (int voxel = 0; voxel < nv; voxel++) {
            auto exponent = exponents[readout][voxel];
            auto factor = factors[readout][voxel];
            auto C = coil_sensitivities[coil][voxel];

            S += exp(exponent * float(sample)) * (factor * C);
        }

        signal[readout][sample][coil] = S;
    }
}

}  // namespace compas