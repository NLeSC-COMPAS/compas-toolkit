
#include "compas/core/complex_type.h"
#include "compas/core/vector.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"
#include "product_kernels.cuh"

namespace compas {
namespace kernels {

// Defined in product_kernels.cuh
__global__ void delta_to_sample_exponent(
        cuda_view_mut<cfloat, 2> E,
        cuda_view_mut<cfloat, 2> dEdT2,
        CartesianTrajectoryView trajectory,
        TissueParametersView parameters);

/// Computes `a + b * conj(c)`
static COMPAS_DEVICE cfloat add_mul_conj(cfloat a, cfloat b, cfloat c) {
    // Writing out the full equation results in better codegen with more FMAs
    return {a.re + b.re * c.re + b.im * c.im, a.im + (-b.re) * c.im + b.im * c.re};
}

template<
        int coils_per_thread = 1,
        int voxels_per_thread=1,
        int readout_tiling_factor=1,
        int sample_tiling_factor=1,
        int threads_per_item=1,
        int threads_per_block=256,
        int blocks_per_sm=5>
__launch_bounds__(threads_per_block, blocks_per_sm) __global__ void jacobian_hermitian_product(
    int nreadouts,
    int nsamples_per_readout,
    int nvoxels,
    int ncoils,
    cfloat* JHv_ptr,
    const cfloat* echos_ptr,
    const cfloat* delta_echos_T1_ptr,
    const cfloat* delta_echos_T2_ptr,
    int parameters_stride,
    const float* parameters_ptr,
    const float* coil_sensitivities_ptr,
    const cfloat* vector_ptr,
    const cfloat* E_ptr,
    const cfloat* dEdT2_ptr) {

    cuda_view_mut<cfloat, 2> JHv = {JHv_ptr, {{4, nvoxels}}};
    cuda_view<cfloat, 2> echos = {echos_ptr, {{nreadouts, nvoxels}}};
    cuda_view<cfloat, 2> delta_echos_T1 = {delta_echos_T1_ptr, {{nreadouts, nvoxels}}};
    cuda_view<cfloat, 2> delta_echos_T2 = {delta_echos_T2_ptr, {{nreadouts, nvoxels}}};
    cuda_view<float, 2> coil_sensitivities = {coil_sensitivities_ptr, {{ncoils, nvoxels}}};
    cuda_view<cfloat, 3> vector = {vector_ptr, {{ncoils, nreadouts, nsamples_per_readout}}};
    cuda_view<cfloat, 2> E = {E_ptr, {{nsamples_per_readout, nvoxels}}};
    cuda_view<cfloat, 2> dEdT2 = {dEdT2_ptr, {{nsamples_per_readout, nvoxels}}};
    cuda_view<float, 2> parameters = {parameters_ptr, {{TissueParameterField::NUM_FIELDS, parameters_stride}}};

    auto ns = nsamples_per_readout;
    auto voxel_offset = index_t(blockIdx.x * blockDim.x + threadIdx.x) / threads_per_item * voxels_per_thread;
    auto lane_id = threads_per_item > 0 ? index_t(threadIdx.x) % threads_per_item : 0;

    if (voxel_offset < nvoxels) {
        cfloat mHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};
        vec2<cfloat> dmHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};

        for (index_t readout_offset = 0; readout_offset < nreadouts; readout_offset += readout_tiling_factor) {
            cfloat me[readout_tiling_factor][voxels_per_thread];
            vec2<cfloat> dme[readout_tiling_factor][voxels_per_thread];

#pragma unroll
            for (index_t r = 0; r < readout_tiling_factor; r++) {
#pragma unroll
                for (index_t v = 0; v < voxels_per_thread; v++) {
                    auto readout = readout_offset + r;
                    auto voxel = voxel_offset + v;

                    me[r][v] = echos[readout][voxel];
                    dme[r][v] = vec2<cfloat>{delta_echos_T1[readout][voxel], delta_echos_T2[readout][voxel]};
                }
            }

#pragma unroll sample_tiling_factor
            for (index_t sample = lane_id; sample < ns; sample += threads_per_item) {
#pragma unroll
                for (index_t v = 0; v < voxels_per_thread; v++) {
#pragma unroll
                    for (index_t r = 0; r < readout_tiling_factor; r++) {
                        auto readout = readout_offset + r;
                        auto voxel = voxel_offset + v;
                        auto Es = E[sample][voxel];

                        auto ms = Es * me[r][v];
                        auto dms = vec2<cfloat>{dme[r][v][0] * Es, dme[r][v][1] * Es + me[r][v] * dEdT2[sample][voxel]};

                        // accumulate dot product in mHv
#pragma unroll
                        for (index_t icoil = 0; icoil < coils_per_thread; icoil++) {
                            auto f = vector[icoil][readout][sample];

                            mHv[v][icoil] = add_mul_conj(mHv[v][icoil], f, ms);
                            dmHv[v][icoil][0] = add_mul_conj(dmHv[v][icoil][0], f, dms[0]);
                            dmHv[v][icoil][1] = add_mul_conj(dmHv[v][icoil][1], f, dms[1]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (index_t v = 0; v < voxels_per_thread; v++) {
            auto voxel = voxel_offset + v;
            // load coordinates, parameters, coil sensitivities and proton density for voxel
            auto T1 = parameters[TissueParameterField::T1][voxel];
            auto T2 = parameters[TissueParameterField::T2][voxel];
            auto rho = cfloat {
                    parameters[TissueParameterField::RHO_X][voxel],
                    parameters[TissueParameterField::RHO_Y][voxel]};

#pragma unroll
            for (index_t i = 0; i < 4; i++) {
                cfloat result = 0.0F;

#pragma unroll
                for (index_t icoil = 0; icoil < coils_per_thread; icoil++) {
                    auto c = coil_sensitivities[icoil][voxel];

                    // size = (nr_nonlinpars + 2) x nr_coils
                    auto tmp = vec4<cfloat>(
                            dmHv[v][icoil][0],
                            dmHv[v][icoil][1],
                            mHv[v][icoil],
                            mHv[v][icoil] * cfloat(0, -1));
                    auto lin_scale = vec4<cfloat>(T1 * c * rho, T2 * c * rho, c, c);

                    result += conj(lin_scale[i]) * tmp[i];
                }

#pragma unroll 6
                for (uint delta = threads_per_item / 2; delta > 0; delta /= 2) {
                    static constexpr uint mask = uint((1L << threads_per_item) - 1);

                    result.re += __shfl_down_sync(mask, result.re, delta, threads_per_item);
                    result.im += __shfl_down_sync(mask, result.im, delta, threads_per_item);
                }

                if (lane_id == 0) {
                    JHv[i][voxel] = result;
                }
            }
        }
    }
}
}  // namespace kernels
}  // namespace compas