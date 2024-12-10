#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/jacobian/product.h"
#include "product_kernels.cuh"

namespace compas {

Array<cfloat, 3> compute_jacobian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 2> vector) {
    static constexpr int threads_per_sample = 16;
    static constexpr dim3 block_dim = {64, 4};
    static constexpr uint threads_per_block = block_dim.x * block_dim.y * block_dim.z;

    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;
    int ncoils = int(coil_sensitivities.size(0));

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos_T1.size(0) == nreadouts);
    COMPAS_ASSERT(delta_echos_T1.size(1) == nvoxels);
    COMPAS_ASSERT(delta_echos_T2.size(0) == nreadouts);
    COMPAS_ASSERT(delta_echos_T2.size(1) == nvoxels);
    COMPAS_ASSERT(coil_sensitivities.size(0) == ncoils);
    COMPAS_ASSERT(coil_sensitivities.size(1) == nvoxels);
    COMPAS_ASSERT(vector.size(0) == 4);  // four reconstruction parameters: T1, T2, rho_x, rho_y
    COMPAS_ASSERT(vector.size(1) == nvoxels);

    auto Jv = Array<cfloat, 3> {{ncoils, nreadouts, ns}};
    auto E = Array<cfloat, 2> {{ns, nvoxels}};
    auto dEdT2 = Array<cfloat, 2> {{ns, nvoxels}};

    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(ns), block_dim.y)};
    ctx.submit_kernel(
        grid_dim,
        block_dim,
        kernels::delta_to_sample_exponent,
        write(E),
        write(dEdT2),
        trajectory,
        parameters);

    // Repeat for each coil
#define COMPAS_COMPUTE_JACOBIAN_IMPL(S, R, C)                                          \
    if (ncoils % (C) == 0 && ns % (S) == 0 && nreadouts % (R) == 0) {                  \
        grid_dim = {                                                                   \
            div_ceil(uint(ns / S * threads_per_sample), block_dim.x),                  \
            div_ceil(uint(nreadouts / R), block_dim.y),                                \
            div_ceil(uint(ncoils / C), block_dim.z)};                                  \
                                                                                       \
        ctx.submit_kernel(                                                             \
            grid_dim,                                                                  \
            block_dim,                                                                 \
            kernels::jacobian_product<threads_per_sample, S, R, C, threads_per_block>, \
            nvoxels,                                                                   \
            nreadouts,                                                                 \
            ns,                                                                        \
            ncoils,                                                                    \
            write(Jv),                                                                 \
            echos,                                                                     \
            delta_echos_T1,                                                            \
            delta_echos_T2,                                                            \
            parameters.parameters.size(1),                                             \
            parameters.parameters,                                                     \
            coil_sensitivities,                                                        \
            E,                                                                         \
            dEdT2,                                                                     \
            vector);                                                                   \
        return Jv;                                                                     \
    }

#define COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(C) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(4, 4, C)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(2, 2, C)         \
    COMPAS_COMPUTE_JACOBIAN_IMPL(1, 1, C)

    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(5)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(4)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(3)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(1)

    throw std::runtime_error("cannot support more than 5 coils");
}

}  // namespace compas