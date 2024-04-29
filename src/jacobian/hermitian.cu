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

    dim3 block_dim = 256;
    dim3 grid_dim = div_ceil(uint(nvoxels), block_dim.x);

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