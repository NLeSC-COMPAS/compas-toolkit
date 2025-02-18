#include "compas/core/vector.h"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

static __global__ void delta_to_sample_exponent(
    kmm::Bounds<2, int> range,
    gpu_subview_mut<cfloat, 2> E,
    gpu_subview_mut<cfloat, 2> dEdT2,
    CartesianTrajectoryView trajectory,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    index_t sample = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (!range.contains(voxel, sample)) {
        return;
    }

    TissueVoxel p = parameters.get(voxel);

    // Read in constants
    auto R2 = 1.0f / p.T2;
    auto ns = trajectory.samples_per_readout;
    auto delta_t = trajectory.delta_t;
    auto delta_k0 = trajectory.delta_k;
    auto x = p.x;
    auto y = p.y;

    // There are ns samples per readout, echo time is assumed to occur
    // at index (ns/2)+1. Now compute sample index relative to the echo time
    float s = float(sample) - 0.5f * float(ns);

    // Apply readout gradient, T₂ decay and B₀ rotation
    auto Theta = delta_k0.re * x + delta_k0.im * y;
    Theta += delta_t * float(2 * M_PI) * p.B0;

    cfloat Es = exp(s * cfloat(-delta_t * R2, Theta));
    cfloat dEsdT2 = (s * delta_t) * R2 * R2 * Es;

    E[sample][voxel] = Es;
    dEdT2[sample][voxel] = dEsdT2;
}

template<
    int threads_per_item = 1,
    int samples_per_thread = 1,
    int readouts_per_thread = 1,
    int coils_per_thread = 1,
    int threads_per_block = 256,
    int blocks_per_sm = 16>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void jacobian_product(
    kmm::Bounds<3, index_t> subrange,
    index_t coil_offset,
    gpu_subview_mut<cfloat, 3> Jv,
    gpu_subview<cfloat, 2> echos,
    gpu_subview<cfloat, 2> delta_echos_T1,
    gpu_subview<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    gpu_subview<cfloat, 2> coil_sensitivities,
    gpu_subview<cfloat, 2> E,
    gpu_subview<cfloat, 2> dEdT2,
    gpu_subview<cfloat, 2> v) {

    index_t voxel_begin = index_t(subrange.x.begin);
    index_t voxel_end = index_t(subrange.x.end);
    index_t sample_offset = index_t(blockIdx.y * blockDim.y + threadIdx.y) * samples_per_thread + subrange.y.begin;
    index_t readout_offset = index_t(blockIdx.z * blockDim.z + threadIdx.z) * readouts_per_thread + subrange.z.begin;
    index_t lane_id = index_t(threadIdx.x);

    if (sample_offset >= subrange.y.end || readout_offset >= subrange.z.end) {
        return;
    }

    cfloat partial_result[samples_per_thread][readouts_per_thread][coils_per_thread];

#pragma unroll
    for (int i = 0; i < samples_per_thread; i++) {
        for (int j = 0; j < readouts_per_thread; j++) {
            for (int k = 0; k < coils_per_thread; k++) {
                partial_result[i][j][k] = cfloat(0);
            }
        }
    }

    for (index_t voxel = voxel_begin + lane_id; voxel < voxel_end; voxel += threads_per_item) {
        // load coordinates, parameters, coil sensitivities and proton density for voxel
        auto p = parameters.get(voxel);
        auto T1 = p.T1;
        auto T2 = p.T2;
        auto rho = p.rho;

#pragma unroll
        for (int i = 0; i < samples_per_thread; i++) {
#pragma unroll
            for (int j = 0; j < readouts_per_thread; j++) {
                int s = sample_offset + i;
                int r = readout_offset + j;

                // load magnetization and partial derivatives at echo time of the r-th readout
                auto me = echos[r][voxel];
                auto dme = vec2<cfloat> {delta_echos_T1[r][voxel], delta_echos_T2[r][voxel]};

                // compute decay (T₂) and rotation (gradients and B₀) to go to sample point
                auto Es = E[s][voxel];
                //            auto dEs = vec2<cfloat> {0, dEdT2[s][voxel]};

                //            auto dm = dme * Es + me * dEs;
                auto dm = vec2<cfloat> {dme[0] * Es, dme[1] * Es + me * dEdT2[s][voxel]};
                auto m = Es * me;

                // store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
                auto dmv = vec4<cfloat>(v[0][voxel], v[1][voxel], v[2][voxel], v[3][voxel]);
                auto lin_scale =
                    vec4<cfloat>(T1 * rho * dm[0], T2 * rho * dm[1], m, m * cfloat(0, 1));
                auto prod = dot(lin_scale, dmv);

#pragma unroll
                for (int k = 0; k < coils_per_thread; k++) {
                    int icoil = coil_offset + k;
                    auto C = coil_sensitivities[icoil][voxel];
                    partial_result[i][j][k] += prod * C;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < samples_per_thread; i++) {
#pragma unroll
        for (int j = 0; j < readouts_per_thread; j++) {
#pragma unroll
            for (int k = 0; k < coils_per_thread; k++) {
                int s = sample_offset + i;
                int r = readout_offset + j;
                int icoil = coil_offset + k;

                cfloat result = partial_result[i][j][k];

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
}
}  // namespace kernels

}  // namespace compas
