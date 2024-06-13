#include "compas/core/utils.h"
#include "compas/core/vector.h"
#include "compas/jacobian/product.h"
#include "hermitian_kernels.cuh"

namespace compas {

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
    auto E = Array<cfloat, 2>(ns, nvoxels);
    auto dEdT2 = Array<cfloat, 2>(ns, nvoxels);

    dim3 block_dim = {64, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(ns), block_dim.y)};
    ctx.submit_kernel(
        grid_dim,
        block_dim,
        kernels::delta_to_sample_exponent,
        write(E),
        write(dEdT2),
        trajectory,
        parameters);

    static constexpr int threads_per_item = 32;
    static constexpr int sample_tiling_factor = 4;

#define COMPAS_COMPUTE_JACOBIAN_IMPL(C, V, R)                                                     \
    if (ncoils == (C) && nvoxels % (V) == 0 && nreadouts % (R) == 0) {                            \
        block_dim = 256;                                                                          \
        grid_dim = div_ceil(uint(nvoxels / (V)) * (threads_per_item), block_dim.x);               \
                                                                                                  \
        ctx.submit_kernel(                                                                        \
            grid_dim,                                                                             \
            block_dim,                                                                            \
            kernels::jacobian_hermitian_product<C, V, R, sample_tiling_factor, threads_per_item>, \
            nreadouts,                                                                            \
            ns,                                                                                   \
            nvoxels,                                                                              \
            ncoils,                                                                               \
            write(JHv),                                                                           \
            echos,                                                                                \
            delta_echos_T1,                                                                       \
            delta_echos_T2,                                                                       \
            parameters.parameters.size(1),                                                        \
            parameters.parameters,                                                                \
            coil_sensitivities,                                                                   \
            vector,                                                                               \
            E,                                                                                    \
            dEdT2);                                                                               \
        return JHv;                                                                               \
    }

#define COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(C) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 2, 8)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 2, 4)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 2, 2)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 2, 1)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 1, 8)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 1, 4)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 1, 2)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 1, 1)

    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(1)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(3)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(4)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(5)

    throw std::runtime_error("cannot support more than 8 coils");
}

}  // namespace compas