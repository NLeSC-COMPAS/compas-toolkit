#include "compas/core/vector.h"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

__global__ void delta_to_sample_exponent(
    cuda_view_mut<cfloat, 2> E,
    cuda_view_mut<cfloat, 2> dEdT2,
    CartesianTrajectoryView trajectory,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    index_t sample = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    TissueVoxel p = parameters.get(voxel);

    // Read in constants
    auto R2 = 1.0f / p.T2;
    auto ns = trajectory.samples_per_readout;
    auto delta_t = trajectory.delta_t;
    auto delta_k0 = trajectory.delta_k;
    auto x = p.x;
    auto y = p.y;

    // There are ns samples per readout, echo time is assumed to occur
    // at index (ns÷2)+1. Now compute sample index relative to the echo time
    float s = float(sample) - 0.5f * float(ns);

    // Apply readout gradient, T₂ decay and B₀ rotation
    auto Theta = delta_k0.re * x + delta_k0.im * y;
    Theta += delta_t * float(2 * M_PI) * p.B0;

    cfloat Es = exp(s * cfloat(-delta_t * R2, Theta));
    cfloat dEsdT2 = (s * delta_t) * R2 * R2 * Es;

    E[sample][voxel] = Es;
    dEdT2[sample][voxel] = dEsdT2;
}

template<int ncoils, int threads_per_item = 1>
__launch_bounds__(256, 16) __global__ void jacobian_product(
    cuda_view_mut<cfloat, 3> Jv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 2> delta_echos_T1,
    cuda_view<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 2> E,
    cuda_view<cfloat, 2> dEdT2,
    cuda_view<cfloat, 2> v) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;

    COMPAS_ASSUME(echos.size(0) == nreadouts);
    COMPAS_ASSUME(echos.size(1) == nvoxels);
    COMPAS_ASSUME(delta_echos_T1.size(0) == nreadouts);
    COMPAS_ASSUME(delta_echos_T1.size(1) == nvoxels);
    COMPAS_ASSUME(delta_echos_T2.size(0) == nreadouts);
    COMPAS_ASSUME(delta_echos_T2.size(1) == nvoxels);
    COMPAS_ASSUME(coil_sensitivities.size(0) == ncoils);
    COMPAS_ASSUME(coil_sensitivities.size(1) == nvoxels);
    COMPAS_ASSUME(v.size(0) == 4);  // four reconstruction parameters: T1, T2, rho_x, rho_y
    COMPAS_ASSUME(v.size(1) == nvoxels);

    index_t s = index_t(blockIdx.x * blockDim.x + threadIdx.x) / threads_per_item;
    index_t r = index_t(blockIdx.y * blockDim.y + threadIdx.y);
    index_t lane_id = threadIdx.x % threads_per_item;
    cfloat partial_result[ncoils];

#pragma unroll
    for (int icoil = 0; icoil < ncoils; icoil++) {
        partial_result[icoil] = cfloat(0);
    }

    if (r < nreadouts && s < ns) {
        for (index_t voxel = lane_id; voxel < nvoxels; voxel += threads_per_item) {
            // load coordinates, parameters, coil sensitivities and proton density for voxel
            auto p = parameters.get(voxel);
            auto rho = p.rho;

            // load magnetization and partial derivatives at echo time of the r-th readout
            auto me = echos[r][voxel];
            auto dme = vec2<cfloat> {delta_echos_T1[r][voxel], delta_echos_T2[r][voxel]};

            // compute decay (T₂) and rotation (gradients and B₀) to go to sample point
            auto Es = E[s][voxel];
            auto dEs = vec2<cfloat> {0, dEdT2[s][voxel]};

            auto dm = dme * Es + me * dEs;
            auto m = Es * me;

            // store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
            auto dmv = vec4<cfloat>(v[0][voxel], v[1][voxel], v[2][voxel], v[3][voxel]);
            auto lin_scale =
                vec4<cfloat>(p.T1 * rho * dm[0], p.T2 * rho * dm[1], m, m * cfloat(0, 1));

#pragma unroll
            for (int icoil = 0; icoil < ncoils; icoil++) {
                auto C = coil_sensitivities[icoil][voxel];
                partial_result[icoil] += dot(lin_scale, dmv) * C;
            }
        }

#pragma unroll
        for (int icoil = 0; icoil < ncoils; icoil++) {
            cfloat result = partial_result[icoil];

#pragma unroll 6
            for (uint delta = threads_per_item / 2; delta > 0; delta /= 2) {
                static constexpr uint mask = uint((1L << threads_per_item) - 1);

                result.re += __shfl_down_sync(mask, result.re, delta, threads_per_item);
                result.im += __shfl_down_sync(mask, result.im, delta, threads_per_item);
            }

            if (lane_id == 0) {
                Jv[icoil][r][s] = result;
            }
        }
    }
}
}  // namespace kernels

}  // namespace compas