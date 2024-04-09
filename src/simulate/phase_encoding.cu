#include "core/utils.h"
#include "trajectories/cartesian_view.cuh"
#include "simulate/phase_encoding.h"
#include "simulate/phase_encoding_kernels.cuh"

namespace compas {

Array<cfloat, 2> phase_encoding(
    CudaContext& ctx,
    Array<cfloat, 2>& echos,
    TissueParameters& parameters,
    Trajectory& trajectory) {
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;

    auto phe_echos = Array<cfloat, 2>(nreadouts, nvoxels);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};
    ctx.submit_kernel(grid_dim, block_dim, kernels::phase_encoding<CartesianTrajectoryView>, write(echos), parameters, trajectory);

    return phe_echos;
}

} // compas
