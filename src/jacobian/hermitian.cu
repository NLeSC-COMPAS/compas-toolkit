#include "core/utils.h"
#include "core/vector.h"
#include "product.h"

namespace compas {

namespace kernels {

/// Computes `a + b * conj(c)`
static __device__ cfloat add_mul_conj(cfloat a, cfloat b, cfloat c) {
    // Writing out the full equation results in better codegen with more FMAs
    return {a.re + b.re * c.re + b.im * c.im, a.im + (-b.re) * c.im + b.im * c.re};
}

template<int ncoils>
static __device__ void expand_readout_and_accumulate_mhv(
    vec<cfloat, ncoils>& mHv,
    vec<vec2<cfloat>, ncoils>& dmHv,
    cfloat me,
    vec2<cfloat> dme,
    TissueVoxel p,
    CartesianTrajectoryView trajectory,
    int readout,
    cuda_view<cfloat, 3> vector) {
    auto ns = trajectory.samples_per_readout;
    auto delta_t = trajectory.delta_t;
    auto delta_k = trajectory.delta_k;
    auto R2 = float(1.0f / p.T2);
    auto x = p.x;
    auto y = p.y;

    // Gradient rotation per sample point
    auto Theta = delta_k.re * x + delta_k.im * y;
    // B₀ rotation per sample point
    Theta += delta_t * p.B0 * float(2 * M_PI);

    //"Rewind" to start of readout
    cfloat R = exp(float(ns) * 0.5f * cfloat(delta_t * R2, -Theta));
    vec2<cfloat> dR = {0, -float(ns) * 0.5f * delta_t * R2 * R2 * R};

    auto dms = dme * R + me * dR;
    auto ms = me * R;

    // T2 decay and gradient- and B₀ induced rotation per sample
    auto E2 = exp(cfloat(-delta_t * R2, Theta));
    auto dE2 = vec2<cfloat> {0, delta_t * R2 * R2 * E2};

#pragma unroll 8
    for (int sample = 0; sample < ns; sample++) {
        // accumulate dot product in mHv
#pragma unroll
        for (int icoil = 0; icoil < ncoils; icoil++) {
            auto v = vector[icoil][readout][sample];

            mHv[icoil] = add_mul_conj(mHv[icoil], v, ms);
            dmHv[icoil][0] = add_mul_conj(dmHv[icoil][0], v, dms[0]);
            dmHv[icoil][1] = add_mul_conj(dmHv[icoil][1], v, dms[1]);
        }

        // compute magnetization at next sample point
        dms = dms * E2 + ms * dE2;
        ms = ms * E2;
    }
}

template<int ncoils = 1>
__global__ void jacobian_hermitian_product(
    cuda_view_mut<cfloat, 2> JHv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 2> delta_echos_T1,
    cuda_view<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 3> vector) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    if (voxel < parameters.nvoxels) {
        int nreadouts = trajectory.nreadouts;
        auto p = parameters.get(voxel);

        vec<cfloat, ncoils> mHv;
        vec<vec2<cfloat>, ncoils> dmHv;

        for (int icoil = 0; icoil < ncoils; icoil++) {
            mHv[icoil] = 0;
            dmHv[icoil] = {0, 0};
        }

        for (int readout = 0; readout < nreadouts; readout++) {
            cfloat me = echos[readout][voxel];
            vec2<cfloat> dme = {delta_echos_T1[readout][voxel], delta_echos_T2[readout][voxel]};

            expand_readout_and_accumulate_mhv(mHv, dmHv, me, dme, p, trajectory, readout, vector);
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            JHv[i][voxel] = 0;
        }

#pragma unroll
        for (int icoil = 0; icoil < ncoils; icoil++) {
            auto c = coil_sensitivities[icoil][voxel];
            auto rho = p.rho;

            // size = (nr_nonlinpars + 2) x nr_coils
            auto tmp = vec4<cfloat>(
                dmHv[icoil][0],
                dmHv[icoil][1],
                mHv[icoil],
                mHv[icoil] * cfloat(0, -1));
            auto lin_scale = vec4<cfloat>(p.T1 * c * rho, p.T2 * c * rho, c, c);

#pragma unroll
            for (int i = 0; i < 4; i++) {
                JHv[i][voxel] += conj(lin_scale[i]) * tmp[i];
            }
        }
    }
}
}  // namespace kernels

Array<cfloat, 2> compute_jacobian_hermitian(
    const CudaContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<float, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int ncoils = coil_sensitivities.size(0);
    int nvoxels = parameters.nvoxels;

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos_T1.size(0) == nreadouts);
    COMPAS_ASSERT(delta_echos_T1.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos_T2.size(0) == nreadouts);
    COMPAS_ASSERT(delta_echos_T2.size(1) == nvoxels);
    COMPAS_ASSERT(coil_sensitivities.size(0) == ncoils);
    COMPAS_ASSERT(coil_sensitivities.size(1) == nvoxels);
    COMPAS_ASSERT(vector.size(0) == ncoils);
    COMPAS_ASSERT(vector.size(1) == nreadouts);
    COMPAS_ASSERT(vector.size(2) == ns);

    // four reconstruction parameters: T1, T2, rho_x, rho_y
    auto JHv = Array<cfloat, 2>(4, nvoxels);

    dim3 block_dim = 256;
    dim3 grid_dim = div_ceil(uint(nreadouts * ns), block_dim.x);

#define COMPAS_COMPUTE_JACOBIAN_IMPL(N)             \
    if (ncoils == (N)) {                            \
        ctx.submit_kernel(                          \
            grid_dim,                               \
            block_dim,                              \
            kernels::jacobian_hermitian_product<N>, \
            write(JHv),                             \
            echos,                                  \
            delta_echos_T1,                         \
            delta_echos_T2,                         \
            parameters,                             \
            trajectory,                             \
            coil_sensitivities,                     \
            vector);                                \
        return JHv;                                 \
    }

    COMPAS_COMPUTE_JACOBIAN_IMPL(1)
    COMPAS_COMPUTE_JACOBIAN_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_IMPL(3)
    COMPAS_COMPUTE_JACOBIAN_IMPL(4)
    COMPAS_COMPUTE_JACOBIAN_IMPL(5)
    COMPAS_COMPUTE_JACOBIAN_IMPL(6)
    COMPAS_COMPUTE_JACOBIAN_IMPL(7)
    COMPAS_COMPUTE_JACOBIAN_IMPL(8)

    throw std::runtime_error("cannot support more than 8 coils");
}

}  // namespace compas