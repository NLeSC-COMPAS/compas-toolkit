#include "core/utils.h"
#include "jacobian/product.h"

namespace compas {

namespace kernels {

struct DeltaMagnetization {
    cfloat m;
    vec2<cfloat> dm;  // In T1 and T2
};

__device__ DeltaMagnetization delta_to_sample_point(
    cfloat m,
    vec2<cfloat> dm,
    CartesianTrajectoryView trajectory,
    int readout_idx,
    int sample_idx,
    TissueVoxel p) {
    // Read in constants
    auto R2 = 1.0f / p.T2;
    auto ns = trajectory.samples_per_readout;
    auto delta_t = trajectory.delta_t;
    auto delta_k0 = trajectory.delta_k;
    auto x = p.x;
    auto y = p.y;

    // There are ns samples per readout, echo time is assumed to occur
    // at index (ns÷2)+1. Now compute sample index relative to the echo time
    float s = float(sample_idx) - 0.5f * float(ns);

    // Apply readout gradient, T₂ decay and B₀ rotation
    auto Theta = delta_k0.re * x + delta_k0.im * y;
    Theta += delta_t * float(2 * M_PI) * p.B0;
    auto E = exp(s * cfloat(-delta_t * R2, Theta));

    auto dE = vec2<cfloat>(0, (s * delta_t) * R2 * R2 * E);

    auto dms = dm * E + m * dE;
    auto ms = E * m;

    return {ms, dms};
}

__global__ void jacobian_product(
    cuda_view_mut<cfloat> Jv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float> coil_sensitivities,
    cuda_view<cfloat, 2> v) {
    auto i = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    int ns = trajectory.samples_per_readout;
    int nr = trajectory.nreadouts;
    auto result = cfloat();

    if (i < nr * ns) {
        int r = i / ns;
        int s = i % ns;
        int nvoxels = parameters.nvoxels;

        for (index_t voxel = 0; voxel < nvoxels; voxel++) {
            // load coordinates, parameters, coil sensitivities and proton density for voxel
            auto p = parameters.get(voxel);
            auto rho = p.rho;

            auto C = coil_sensitivities[voxel];

            // load magnetization and partial derivatives at echo time of the r-th readout
            auto me = echos[r][voxel];
            auto dme = vec2<cfloat> {delta_echos[0][r][voxel], delta_echos[1][r][voxel]};

            // compute decay (T₂) and rotation (gradients and B₀) to go to sample point
            auto [m, dm] = delta_to_sample_point(me, dme, trajectory, r, s, p);

            // store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
            auto dmv = vec4<cfloat>(v[0][voxel], v[1][voxel], v[2][voxel], v[3][voxel])
                * vec4<cfloat>(dm[0], dm[1], m, m * cfloat(0, 1));

            auto lin_scale = vec4<cfloat>(p.T1 * C * rho, p.T2 * C * rho, C, C);

            result += dot(lin_scale, dmv);
        }

        Jv[i] = result;
    }
}
}  // namespace kernels

void compute_jacobian(
    const CudaContext& ctx,
    cuda_view_mut<cfloat, 2> Jv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 2> vector) {
    CudaContextGuard guard {ctx};

    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;
    int ncoils = coil_sensitivities.size(0);

    COMPAS_ASSERT(Jv.size(0) == ncoils);
    COMPAS_ASSERT(Jv.size(1) == nreadouts * ns);
    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos.size(0) == 2);  // T1 and T2
    COMPAS_ASSERT(delta_echos.size(1) == nreadouts);
    COMPAS_ASSERT(delta_echos.size(2) == nvoxels);
    COMPAS_ASSERT(coil_sensitivities.size(0) == ncoils);
    COMPAS_ASSERT(coil_sensitivities.size(1) == nvoxels);
    COMPAS_ASSERT(vector.size(0) == 4);  // four reconstruction parameters: T1, T2, rho_x, rho_y
    COMPAS_ASSERT(vector.size(1) == nvoxels);

    dim3 block_dim = 256;
    dim3 grid_dim = div_ceil(uint(nreadouts * ns), block_dim.x);

    // Repeat for each coil
    for (int icoil = 0; icoil < ncoils; icoil++) {
        kernels::jacobian_product<<<grid_dim, block_dim>>>(
            Jv.drop_leading_axis(icoil),
            echos,
            delta_echos,
            parameters,
            trajectory,
            coil_sensitivities.drop_leading_axis(icoil),
            vector);
    }
}
}  // namespace compas