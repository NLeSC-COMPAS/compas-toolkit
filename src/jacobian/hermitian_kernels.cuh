
#include "compas/core/complex_type.h"
#include "compas/core/vector.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"
#include "product_kernels.cuh"

namespace compas {
namespace kernels {

// Defined in product_kernels.cuh
static __global__ void compute_sample_decay(
    kmm::Bounds<2, int> range,
    GPUSubviewMut<cfloat, 2> E,
    GPUSubviewMut<cfloat, 2> dEdT2,
    CartesianTrajectoryView trajectory,
    TissueParametersView parameters);

/// Computes `a + b * conj(c)`
__device__ static COMPAS_DEVICE cfloat add_mul_conj(cfloat a, cfloat b, cfloat c) {
    // Writing out the full equation results in better codegen with more FMAs
    return {fmaf(b.im, c.im, fmaf(b.re, c.re, a.re)), fmaf(b.im, c.re, fmaf(-b.re, c.im, a.im))};
}

template<typename T, typename U>
__global__ void
convert_kernel(kmm::Bounds<3, int> subrange, GPUSubviewMut<T, 3> output, GPUSubview<U, 3> input) {
    auto k = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto j = index_t(blockIdx.y * blockDim.y + threadIdx.y);
    auto i = index_t(blockIdx.z * blockDim.z + threadIdx.z);

    if (input.in_bounds({i, j, k})) {
        output[i][j][k] = T {input[i][j][k]};
    }
}

template<
    int voxel_tile_size = 1,
    int readout_tile_size = 1,
    int sample_tile_size = 1,
    int coils_per_thread = 1,
    int block_size_x = 64,
    int block_size_y = 1,
    int block_size_z = 1,
    int blocks_per_sm = 1,
    bool use_smem = true>
__launch_bounds__(block_size_x* block_size_y* block_size_z, blocks_per_sm) __global__
    void jacobian_hermitian_product(
        kmm::Bounds<3, int> subrange,
        GPUSubviewMut<cfloat, 2> JHv,
        GPUSubview<cfloat, 2> echos,
        GPUSubview<cfloat, 2> delta_echos_T1,
        GPUSubview<cfloat, 2> delta_echos_T2,
        TissueParametersView parameters,
        GPUSubview<cfloat, 2> coil_sensitivities,
        GPUSubview<cfloat, 3> vector,
        GPUSubview<cfloat, 2> E,
        GPUSubview<cfloat, 2> dEdT2) {
    static constexpr index_t voxels_per_thread =
        (voxel_tile_size / block_size_x) + int(voxel_tile_size % block_size_x > 0);
    static constexpr index_t readouts_per_thread = readout_tile_size / block_size_y;
    static constexpr index_t samples_per_thread = sample_tile_size / block_size_z;

    auto tid_x = block_size_x > 0 ? index_t(threadIdx.x) : 0;
    auto tid_y = block_size_y > 0 ? index_t(threadIdx.y) : 0;
    auto tid_z = block_size_z > 0 ? index_t(threadIdx.z) : 0;

    auto voxel_block_begin = index_t(blockIdx.x * voxel_tile_size + subrange.x.begin);
    auto voxel_block_end =
        min(index_t(voxel_block_begin + voxel_tile_size), index_t(subrange.x.end));

    __builtin_assume(tid_x >= 0 && tid_x < block_size_x);
    __builtin_assume(tid_y >= 0 && tid_y < block_size_y);
    __builtin_assume(tid_z >= 0 && tid_z < block_size_z);

    __shared__ float shared_reduction[2][block_size_z][block_size_y][block_size_x];
    __shared__ float shared_E[2][samples_per_thread][block_size_z][block_size_x];
    __shared__ float shared_dEdT2[2][samples_per_thread][block_size_z][block_size_x];

    cfloat mHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};
    vec2<cfloat> dmHv[voxels_per_thread][coils_per_thread] = {cfloat(0)};

    for (index_t lv = 0; lv < voxels_per_thread; lv++) {
        index_t voxel = voxel_block_begin + lv * block_size_x + tid_x;

        if (voxel >= voxel_block_end) {
            break;
        }

        for (index_t sample_offset = subrange.y.begin; sample_offset < subrange.y.end;
             sample_offset += sample_tile_size) {
            cfloat local_E[samples_per_thread];
            cfloat local_dEdT2[samples_per_thread];

            if (use_smem) {
                __syncthreads();
            }

#pragma unroll samples_per_thread
            for (index_t s = 0; s < samples_per_thread; s++) {
                auto sample = sample_offset + s * block_size_z + tid_z;

                if (use_smem) {
                    if (tid_y == 0) {
                        shared_E[0][s][tid_z][tid_x] = E[sample][voxel].real();
                        shared_E[1][s][tid_z][tid_x] = E[sample][voxel].imag();

                        shared_dEdT2[0][s][tid_z][tid_x] = dEdT2[sample][voxel].real();
                        shared_dEdT2[1][s][tid_z][tid_x] = dEdT2[sample][voxel].imag();
                    }
                } else {
                    local_E[s] = E[sample][voxel];
                    local_dEdT2[s] = dEdT2[sample][voxel];
                }
            }

            if (use_smem) {
                __syncthreads();
            }

            for (index_t readout_offset = subrange.z.begin; readout_offset < subrange.z.end;
                 readout_offset += readout_tile_size) {
#pragma unroll readouts_per_thread
                for (index_t r = 0; r < readouts_per_thread; r++) {
                    auto readout = readout_offset + r * block_size_y + tid_y;
                    cfloat me = echos[readout][voxel];
                    vec2<cfloat> dme = vec2<cfloat> {
                        delta_echos_T1[readout][voxel],
                        delta_echos_T2[readout][voxel]};

#pragma unroll samples_per_thread
                    for (index_t s = 0; s < samples_per_thread; s++) {
                        auto sample = sample_offset + s * block_size_z + tid_z;
                        cfloat Es;
                        cfloat dET2s;

                        if (use_smem) {
                            Es =
                                cfloat {shared_E[0][s][tid_z][tid_x], shared_E[1][s][tid_z][tid_x]};
                            dET2s = cfloat {
                                shared_dEdT2[0][s][tid_z][tid_x],
                                shared_dEdT2[1][s][tid_z][tid_x]};
                        } else {
                            Es = local_E[s];
                            dET2s = local_dEdT2[s];
                        }

                        auto ms = Es * me;
                        auto dms = vec2<cfloat> {dme[0] * Es, dme[1] * Es + me * dET2s};

#pragma unroll coils_per_thread
                        for (index_t icoil = 0; icoil < coils_per_thread; icoil++) {
                            auto f = cfloat {vector[icoil][readout][sample]};

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
        index_t voxel = voxel_block_begin + lv * block_size_x + tid_x;

        if (voxel >= voxel_block_end) {
            break;
        }

        // load coordinates, parameters, coil sensitivities and proton density for voxel
        auto p = parameters.get(voxel);
        auto T1 = p.T1;
        auto T2 = p.T2;
        auto rho = p.rho;

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
