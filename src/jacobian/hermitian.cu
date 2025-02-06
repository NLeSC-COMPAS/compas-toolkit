#include "compas/core/utils.h"
#include "compas/core/vector.h"
#include "compas/jacobian/product.h"
#include "hermitian_kernels.cuh"

namespace compas {

void launch_jacobian_hermitian_kernel(
    kmm::DeviceContext& ctx,
    kmm::NDRange subrange,
    gpu_subview_mut<cfloat, 2> JHv,
    gpu_subview<cfloat, 2> echos,
    gpu_subview<cfloat, 2> delta_echos_T1,
    gpu_subview<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    gpu_subview<cfloat, 2> coil_sensitivities,
    gpu_subview<cfloat, 3> vector,
    gpu_subview<cfloat, 2> E,
    gpu_subview<cfloat, 2> dEdT2) {
    uint ncoils = coil_sensitivities.size(0);
    uint nvoxels = subrange.x.size();
    uint ns = subrange.y.size();
    uint nreadouts = subrange.z.size();

#define COMPAS_COMPUTE_JACOBIAN_IMPL(C, V, R, S, BX, BY, BZ)        \
    if (ncoils == (C) && nreadouts % (R) == 0 && ns % (S) == 0) {   \
        dim3 block_size = {BX, BY, BZ};                             \
        dim3 grid_size = {div_ceil(nvoxels, uint(V))};              \
                                                                    \
        kernels::jacobian_hermitian_product<V, R, S, C, BX, BY, BZ> \
            <<<grid_size, block_size, 0, ctx>>>(                    \
                subrange,                                           \
                JHv,                                                \
                echos,                                              \
                delta_echos_T1,                                     \
                delta_echos_T2,                                     \
                parameters,                                         \
                coil_sensitivities,                                 \
                vector,                                             \
                E,                                                  \
                dEdT2);                                             \
        return;                                                     \
    }

#define COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(C)        \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 32, 32, 2, 2) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 16, 32, 2, 2) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 8, 8, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 4, 4, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 2, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 128, 1, 1, 128, 1, 1)

    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(1)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(3)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(4)

    throw std::runtime_error("cannot support more than 4 coils");
}

Array<cfloat, 2> compute_jacobian_hermitian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    using namespace kmm::placeholders;

    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int ncoils = int(coil_sensitivities.size(0));
    int nvoxels = parameters.nvoxels;
    int chunk_size = parameters.chunk_size;

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
    auto JHv = Array<cfloat, 2> {{4, nvoxels}};
    auto E = Array<cfloat, 2> {{ns, nvoxels}};
    auto dEdT2 = Array<cfloat, 2> {{ns, nvoxels}};
    auto _voxel = kmm::Axis(0);

    dim3 block_dim = {64, 4};
    ctx.parallel_submit(
        {nvoxels, ns},
        {chunk_size, ns},
        kmm::GPUKernel(kernels::delta_to_sample_exponent, block_dim),
        write(E(_, _voxel)),
        write(dEdT2(_, _voxel)),
        trajectory,
        read(parameters, _voxel));

    ctx.parallel_submit(
        {nvoxels, ns, nreadouts},
        {chunk_size, ns, nreadouts},
        kmm::GPU(launch_jacobian_hermitian_kernel),
        write(JHv(_, _voxel)),
        echos(_, _voxel),
        delta_echos_T1(_, _voxel),
        delta_echos_T2(_, _voxel),
        read(parameters, _voxel),
        coil_sensitivities(_, _voxel),
        vector(_, _, _),
        E(_, _voxel),
        dEdT2(_, _voxel));

    return JHv;
}

}  // namespace compas