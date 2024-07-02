
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
__device__
static COMPAS_DEVICE cfloat add_mul_conj(cfloat a, cfloat b, cfloat c) {
    // Writing out the full equation results in better codegen with more FMAs
    return {
        fmaf(b.im, c.im, fmaf(b.re, c.re, a.re)),
        fmaf(b.im, c.re, fmaf(-b.re, c.im, a.im))
    };
}

template<
        int coils_per_thread = 1,
        int voxel_tile_size=1,
        int readout_tile_size=1,
        int sample_tile_size=1,
        int block_size_x=64,
        int block_size_y=1,
        int block_size_z=1,
        int blocks_per_sm=1,
        bool smem_vector=false,
        bool smem_echos=false>
__launch_bounds__(block_size_x*block_size_y*block_size_z, blocks_per_sm) __global__ void jacobian_hermitian_product(
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

    auto tid_x = block_size_x > 0 ? index_t(threadIdx.x) : 0;
    auto tid_y = block_size_y > 0 ? index_t(threadIdx.y) : 0;
    auto tid_z = block_size_z > 0 ? index_t(threadIdx.z) : 0;
    auto voxel_block_offset = index_t(blockIdx.x * voxel_tile_size);

    __builtin_assume(tid_x >= 0 && tid_x < block_size_x);
    __builtin_assume(tid_y >= 0 && tid_y < block_size_y);
    __builtin_assume(tid_z >= 0 && tid_z < block_size_z);

    __shared__ float shared_reduction[2][block_size_z][block_size_y][block_size_x];
    __shared__ float shared_vector[2][coils_per_thread][readout_tile_size][sample_tile_size];
    __shared__ float shared_echos[2][readout_tile_size][voxel_tile_size];
    __shared__ float shared_dechos[4][readout_tile_size][voxel_tile_size];

    if (voxel_block_offset >= nvoxels) {
        return;
    }

    static constexpr index_t voxels_per_thread =
            (voxel_tile_size / block_size_x) + int(voxel_tile_size % block_size_x > 0);

    cfloat mHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};
    vec2<cfloat> dmHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};

    for (index_t readout_offset = 0; readout_offset < nreadouts; readout_offset += readout_tile_size) {
        if (smem_echos) {
            __syncthreads();

            if (tid_z == 0) {
#pragma unroll voxels_per_thread
                for (index_t v = tid_x; v < voxel_tile_size; v += block_size_x) {
                    index_t voxel = voxel_block_offset + v;
                    if (voxel >= nvoxels) {
                        break;
                    }

#pragma unroll readout_tile_size / block_size_y
                    for (index_t r = tid_y; r < readout_tile_size; r += block_size_y) {
                        auto readout = readout_offset + r;

                        shared_echos[0][r][v] = echos[readout][voxel].re;
                        shared_echos[1][r][v] = echos[readout][voxel].im;

                        shared_dechos[0][r][v] = delta_echos_T1[readout][voxel].re;
                        shared_dechos[1][r][v] = delta_echos_T1[readout][voxel].im;

                        shared_dechos[2][r][v] = delta_echos_T2[readout][voxel].re;
                        shared_dechos[3][r][v] = delta_echos_T2[readout][voxel].im;
                    }
                }
            }

            __syncthreads();
        }

        for (index_t sample_offset = 0; sample_offset < nsamples_per_readout; sample_offset += sample_tile_size) {
            if (smem_vector) {
                __syncthreads();

#pragma unroll
                for (int icoil = tid_z; icoil < coils_per_thread; icoil += block_size_z) {
#pragma unroll
                    for (int r = tid_y; r < readout_tile_size; r += block_size_y) {
#pragma unroll
                        for (int s = tid_x; s < sample_tile_size; s += block_size_x) {
                            auto readout = readout_offset + r;
                            auto sample = sample_offset + s;

                            shared_vector[0][icoil][r][s] = vector[icoil][readout][sample].re;
                            shared_vector[1][icoil][r][s] = vector[icoil][readout][sample].im;
                        }
                    }
                }

                __syncthreads();
            }

#pragma unroll voxels_per_thread
            for (index_t lv = 0; lv < voxels_per_thread; lv++) {
                index_t v = lv * block_size_x + tid_x;
                index_t voxel = voxel_block_offset + v;

                if (voxel >= nvoxels || v >= voxel_tile_size) {
                    break;
                }

#pragma unroll readout_tile_size / block_size_y
                for (index_t r = tid_y; r < readout_tile_size; r += block_size_y) {
                    auto readout = readout_offset + r;
                    cfloat me;
                    vec2<cfloat> dme;

                    if (smem_echos) {
                        me = cfloat(shared_echos[0][r][v], shared_echos[1][r][v]);
                        dme[0] = cfloat(shared_dechos[0][r][v], shared_dechos[1][r][v]);
                        dme[1] = cfloat(shared_dechos[2][r][v], shared_dechos[3][r][v]);
                    } else {
                        me = echos[readout][voxel];
                        dme = vec2<cfloat>{delta_echos_T1[readout][voxel], delta_echos_T2[readout][voxel]};
                    }

#pragma unroll sample_tile_size / block_size_z
                    for (index_t s = tid_z; s < sample_tile_size; s += block_size_z) {
                        auto sample = sample_offset + s;
                        auto Es = E[sample][voxel];

                        auto ms = Es * me;
                        auto dms = vec2<cfloat>{dme[0] * Es, dme[1] * Es + me * dEdT2[sample][voxel]};

#pragma unroll coils_per_thread
                        for (index_t icoil = 0; icoil < coils_per_thread; icoil++) {
                            auto f = smem_vector ?
                                     cfloat(shared_vector[0][icoil][r][s], shared_vector[1][icoil][r][s]) :
                                     vector[icoil][readout][sample];

                            // accumulate dot product in mHv
                            mHv[lv][icoil] = add_mul_conj(mHv[lv][icoil], f, ms);
                            dmHv[lv][icoil][0] = add_mul_conj(dmHv[lv][icoil][0], f, dms[0]);
                            dmHv[lv][icoil][1] = add_mul_conj(dmHv[lv][icoil][1], f, dms[1]);
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (index_t lv = 0; lv < voxels_per_thread; lv++) {
        index_t v = lv * block_size_x + tid_x;
        index_t voxel = voxel_block_offset + v;

        if (voxel >= nvoxels || v >= voxel_tile_size) {
            break;
        }

        // load coordinates, parameters, coil sensitivities and proton density for voxel
        auto T1 = parameters[TissueParameterField::T1][voxel];
        auto T2 = parameters[TissueParameterField::T2][voxel];
        auto rho = cfloat{
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
                        dmHv[lv][icoil][0],
                        dmHv[lv][icoil][1],
                        mHv[lv][icoil],
                        mHv[lv][icoil] * cfloat(0, -1));
                auto lin_scale = vec4<cfloat>(T1 * c * rho, T2 * c * rho, c, c);

                result += conj(lin_scale[i]) * tmp[i];
            }

            if (block_size_y > 0 || block_size_z > 0) {
                __syncthreads();
                shared_reduction[0][tid_z][tid_y][tid_x] = result.re;
                shared_reduction[1][tid_z][tid_y][tid_x] = result.im;
                __syncthreads();

                if (tid_y == 0 && tid_z == 0) {
#pragma unroll
                    for (index_t y = 0; y < block_size_y; y++) {
#pragma unroll
                        for (index_t z = 0; z < block_size_z; z++) {
                            if (y != 0 || z != 0) {
                                result.re += shared_reduction[0][z][y][tid_x];
                                result.im += shared_reduction[1][z][y][tid_x];
                            }
                        }
                    }
                }
            }

            if (tid_y == 0 && tid_z == 0) {
                JHv[i][voxel] = result;
            }
        }
    }
}
}  // namespace kernels
}  // namespace compas