#include "compas/core/utils.h"
#include "compas/trajectories/phase_encoding.h"
#include "phase_encoding_kernels.cuh"

namespace compas {

Array<cfloat, 2> phase_encoding(
    const CudaContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const CartesianTrajectory& trajectory) {
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;

    auto ph_en_echos = Array<cfloat, 2>(nreadouts, nvoxels);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};
    ctx.submit_kernel(
        grid_dim,
        block_dim,
        kernels::phase_encoding,
        write(ph_en_echos),
        echos,
        parameters,
        trajectory);

    return ph_en_echos;
}

}  // namespace compas
