
#include "compas/core/complex_type.h"
#include "compas/core/vector.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/trajectories/cartesian_view.cuh"

namespace compas {
namespace kernels {

/// Computes `a + b * conj(c)`
static COMPAS_DEVICE cfloat add_mul_conj(cfloat a, cfloat b, cfloat c) {
    // Writing out the full equation results in better codegen with more FMAs
    return {a.re + b.re * c.re + b.im * c.im, a.im + (-b.re) * c.im + b.im * c.re};
}

template<int ncoils>
static COMPAS_DEVICE void expand_readout_and_accumulate_mhv(
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
}  // namespace compas