#include "compas/core/vector.h"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

static __global__ void jacobian_product_naive(
    kmm::Range<int> voxels,
    int icoil,
    int nreadouts,
    int ns,
    GPUSubviewMut<cfloat, 3> Jv,
    GPUSubview<cfloat, 2> echos,
    GPUSubview<cfloat, 2> delta_echos_T1,
    GPUSubview<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubview<cfloat, 2> vector) {
    index_t voxel_begin = index_t(voxels.begin);
    index_t voxel_end = index_t(voxels.end);
    index_t s = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    index_t r = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (s >= ns || r >= nreadouts) {
        return;
    }

    cfloat result = 0;

    for (index_t voxel = voxel_begin; voxel < voxel_end; voxel++) {
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
        float sample = float(s) - 0.5f * float(ns);

        // Apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k0.re * x + delta_k0.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        cfloat Es = exp(sample * cfloat(-delta_t * R2, Theta));
        cfloat dEsdT2 = (sample * delta_t) * R2 * R2 * Es;

        auto me = echos[r][voxel];
        auto dme = vec2<cfloat> {delta_echos_T1[r][voxel], delta_echos_T2[r][voxel]};

        auto prod =  //
            vector[0][voxel] * Es * p.T1 * p.rho * dme[0] +  //
            vector[1][voxel] * (Es * dme[1] + dEsdT2 * me) * p.T2 * p.rho +  //
            vector[2][voxel] * Es * me +  //
            vector[3][voxel] * Es * cfloat(0, 1) * me;  //

        auto C = coil_sensitivities[icoil][voxel];
        result += prod * C;
    }

    Jv[icoil][r][s] = result;
}

static __global__ void compute_sample_decay(
    kmm::Bounds<2, int> range,
    GPUSubviewMut<cfloat, 2> E,
    GPUSubviewMut<cfloat, 2> dEdT2,
    CartesianTrajectoryView trajectory,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    index_t sample = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (!range.contains(voxel, sample)) {
        return;
    }

    TissueVoxel p = parameters.get(voxel);
    E[sample][voxel] = trajectory.calculate_sample_decay_absolute(sample, p);
    dEdT2[sample][voxel] = trajectory.calculate_sample_decay_absolute_delta_T2(sample, p);
}

template<typename T = float>
static __global__ void compute_sample_decay_planar(
    kmm::Bounds<2, int> range,
    GPUSubviewMut<T, 3> E,
    GPUSubviewMut<T, 3> dEdT2,
    CartesianTrajectoryView trajectory,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    index_t sample = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (!range.contains(voxel, sample)) {
        return;
    }

    TissueVoxel p = parameters.get(voxel);

    auto v = trajectory.calculate_sample_decay_absolute(sample, p);
    E[0][sample][voxel] = kernel_float::cast<T>(v.re);
    E[1][sample][voxel] = kernel_float::cast<T>(v.im);

    auto dv = trajectory.calculate_sample_decay_absolute_delta_T2(sample, p);
    dEdT2[0][sample][voxel] = kernel_float::cast<T>(dv.re);
    dEdT2[1][sample][voxel] = kernel_float::cast<T>(dv.im);
}

static __global__ void compute_adjoint_sources(
    kmm::Bounds<2, int> range,
    GPUSubviewMut<cfloat, 2> adj_phase,
    GPUSubviewMut<cfloat, 2> adj_decay,
    GPUSubview<cfloat, 2> echos,
    GPUSubview<cfloat, 2> delta_echos_T1,
    GPUSubview<cfloat, 2> delta_echos_T2,
    GPUSubview<cfloat, 2> vector,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    index_t readout = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (!range.contains(voxel, readout)) {
        return;
    }

    auto p = parameters.get(voxel);
    auto me = echos[readout][voxel];
    auto dme = vec2<cfloat> {delta_echos_T1[readout][voxel], delta_echos_T2[readout][voxel]};

    adj_phase[readout][voxel] =  //
        vector[0][voxel] * p.T1 * p.rho * dme[0] +  //
        vector[1][voxel] * p.T2 * p.rho * dme[1] +  //
        vector[2][voxel] * me +  //
        vector[3][voxel] * cfloat(0, 1) * me;

    adj_decay[readout][voxel] = vector[1][voxel] * p.T2 * p.rho * me;
}

template<typename T>
static __global__ void compute_adjoint_sources_with_coil(
    kmm::Bounds<2, int> range,
    GPUSubviewMut<T, 3> adj_phase,  // planar complex
    GPUSubviewMut<T, 3> adj_decay,  // planar complex
    int icoil,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubview<cfloat, 2> echos,
    GPUSubview<cfloat, 2> delta_echos_T1,
    GPUSubview<cfloat, 2> delta_echos_T2,
    GPUSubview<cfloat, 2> vector,
    TissueParametersView parameters) {
    index_t voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x + range.x.begin);
    index_t readout = index_t(blockIdx.y * blockDim.y + threadIdx.y + range.y.begin);

    if (!range.contains(voxel, readout)) {
        return;
    }

    auto p = parameters.get(voxel);
    auto me = echos[readout][voxel];
    auto dme = vec2<cfloat> {delta_echos_T1[readout][voxel], delta_echos_T2[readout][voxel]};
    auto C = coil_sensitivities[icoil][voxel];

    auto phase =  //
        vector[0][voxel] * p.T1 * p.rho * dme[0] +  //
        vector[1][voxel] * p.T2 * p.rho * dme[1] +  //
        vector[2][voxel] * me +  //
        vector[3][voxel] * cfloat(0, 1) * me;

    cfloat v = C * phase;
    adj_phase[0][readout][voxel] = kernel_float::cast<T>(v.re);
    adj_phase[1][readout][voxel] = kernel_float::cast<T>(v.im);

    cfloat dv = C * vector[1][voxel] * p.T2 * p.rho * me;
    adj_decay[0][readout][voxel] = kernel_float::cast<T>(dv.re);
    adj_decay[1][readout][voxel] = kernel_float::cast<T>(dv.im);
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
    GPUSubviewMut<cfloat, 3> Jv,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubview<cfloat, 2> E,
    GPUSubview<cfloat, 2> dEdT2,
    GPUSubview<cfloat, 2> adj_phase,
    GPUSubview<cfloat, 2> adj_decay) {
    index_t voxel_begin = index_t(subrange.x.begin);
    index_t voxel_end = index_t(subrange.x.end);
    index_t sample_offset =
        index_t(blockIdx.y * blockDim.y + threadIdx.y) * samples_per_thread + subrange.y.begin;
    index_t readout_offset =
        index_t(blockIdx.z * blockDim.z + threadIdx.z) * readouts_per_thread + subrange.z.begin;
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
#pragma unroll
        for (int i = 0; i < samples_per_thread; i++) {
#pragma unroll
            for (int j = 0; j < readouts_per_thread; j++) {
                int s = sample_offset + i;
                int r = readout_offset + j;

                auto prod =  //
                    adj_phase[r][voxel] * E[s][voxel] +  //
                    adj_decay[r][voxel] * dEdT2[s][voxel];

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
#ifdef COMPAS_IS_CUDA
                    static constexpr uint mask = uint((1L << threads_per_item) - 1);
#elif defined(COMPAS_IS_HIP)
                    static constexpr long long unsigned int mask =
                        uint((1L << threads_per_item) - 1);
#endif

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
