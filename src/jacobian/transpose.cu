#include "core/utils.h"
#include "product.h"

namespace compas {

namespace kernels {

static __device__ void expand_readout_and_accumulate_mhv(
    cfloat& mHv,
    vec2<cfloat>& dmHv,
    cfloat me,
    vec2<cfloat> dme,
    TissueVoxel p,
    CartesianTrajectoryView trajectory,
    int readout,
    cuda_view<cfloat> vector) {
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

    for (int sample = 0; sample < ns; sample++) {
        index_t t = readout * ns + sample;

        // accumulate dot product in mHv
        auto v = vector[t];
        mHv += conj(ms) * v;
        dmHv[0] += conj(dms[0]) * v;
        dmHv[1] += conj(dms[1]) * v;

        // compute magnetization at next sample point
        dms = dms * E2 + ms * dE2;
        ms = ms * E2;
    }
}

__global__ void jacobian_transposed_product(
    cuda_view_mut<cfloat, 2> JHv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float> coil_sensitivities,
    cuda_view<cfloat> vector) {
    auto voxel = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    if (voxel < parameters.nvoxels) {
        int nreadouts = trajectory.nreadouts;
        auto p = parameters.get(voxel);

        cfloat mHv = 0;
        vec2<cfloat> dmHv = {0, 0};

        for (int readout = 0; readout < nreadouts; readout++) {
            cfloat me = echos[readout][voxel];
            vec2<cfloat> dme = {delta_echos[0][readout][voxel], delta_echos[1][readout][voxel]};

            expand_readout_and_accumulate_mhv(mHv, dmHv, me, dme, p, trajectory, readout, vector);
        }

        auto c = coil_sensitivities[voxel];
        auto rho = p.rho;

        // size = (nr_nonlinpars + 2) x nr_coils
        auto tmp = vec4<cfloat>(dmHv[0], dmHv[1], mHv, mHv * cfloat(0, -1));
        auto lin_scale = vec4<cfloat>(p.T1 * c * rho, p.T2 * c * rho, c, c);

        for (int i = 0; i < 4; i++) {
            JHv[i][voxel] += conj(lin_scale[i]) * tmp[i];
        }
    }
}
}  // namespace kernels

void compute_jacobian_transposed(
    const CudaContext& ctx,
    cuda_view_mut<cfloat, 2> JHv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 2> vector) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int ncoils = coil_sensitivities.size(0);
    int nvoxels = parameters.nvoxels;

    COMPAS_ASSERT(JHv.size(0) == 4);  // four reconstruction parameters: T1, T2, rho_x, rho_y
    COMPAS_ASSERT(JHv.size(1) == nvoxels);
    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos.size(0) == 2);  // T1 and T2
    COMPAS_ASSERT(delta_echos.size(1) == nreadouts);
    COMPAS_ASSERT(delta_echos.size(2) == nvoxels);
    COMPAS_ASSERT(coil_sensitivities.size(0) == ncoils);
    COMPAS_ASSERT(coil_sensitivities.size(1) == nvoxels);
    COMPAS_ASSERT(vector.size(0) == ncoils);
    COMPAS_ASSERT(vector.size(1) == nreadouts * ns);

    ctx.fill(JHv, cfloat());

    for (int icoil = 0; icoil < ncoils; icoil++) {
        dim3 block_dim = 256;
        dim3 grid_dim = div_ceil(uint(nreadouts * ns), block_dim.x);

        kernels::jacobian_transposed_product<<<grid_dim, block_dim>>>(
            JHv,
            echos,
            delta_echos,
            parameters,
            trajectory,
            coil_sensitivities.drop_leading_axis(icoil),
            vector.drop_leading_axis(icoil));
    }
}

}  // namespace compas