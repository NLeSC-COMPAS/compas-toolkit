#include "compas/core/utils.h"
#include "compas/core/vector.h"
#include "compas/jacobian/product.h"
#include "hermitian_kernels.cuh"

namespace compas {

Array<cfloat, 2> compute_jacobian_hermitian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int ncoils = int(coil_sensitivities.size(0));
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
    auto JHv = Array<cfloat, 2> {{4, nvoxels}};
    auto E = Array<cfloat, 2> {{ns, nvoxels}};
    auto dEdT2 = Array<cfloat, 2> {{ns, nvoxels}};

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

    auto vector_lo = Array<complex_type<__nv_bfloat16>, 3> {{ncoils, nreadouts, ns}};
    grid_dim = {uint(div_ceil(ns, 16)), uint(div_ceil(nreadouts, 16)), uint(ncoils)};

    ctx.submit_kernel(
        grid_dim,
        {16, 16},
        kernels::convert_kernel<complex_type<__nv_bfloat16>, complex_type<float>>,
        write(vector_lo),
        vector);

#define COMPAS_COMPUTE_JACOBIAN_IMPL(C, V, R, S, BX, BY, BZ)                \
    if (ncoils == (C) && nreadouts % (R) == 0 && ns % (S) == 0) {           \
        block_dim = {BX, BY, BZ};                                           \
        grid_dim = div_ceil(uint(nvoxels), uint(V));                        \
                                                                            \
        ctx.submit_kernel(                                                  \
            grid_dim,                                                       \
            block_dim,                                                      \
            kernels::jacobian_hermitian_product<C, V, R, S, BX, BY, BZ, 1>, \
            nreadouts,                                                      \
            ns,                                                             \
            nvoxels,                                                        \
            ncoils,                                                         \
            write(JHv),                                                     \
            echos,                                                          \
            delta_echos_T1,                                                 \
            delta_echos_T2,                                                 \
            parameters.parameters.size(1),                                  \
            parameters.parameters,                                          \
            coil_sensitivities,                                             \
            vector,                                                         \
            E,                                                              \
            dEdT2);                                                         \
        return JHv;                                                         \
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

}  // namespace compas