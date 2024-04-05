#include "simulate/phase_encoding.h"
#include "simulate/phase_encoding_kernels.cuh"

namespace compas {

Array<cfloat, 2> phase_encoding(
    CudaContext ctx,
    Array<cfloat, 2>& echos,
    TissueParameters& parameters,
    Trajectory& trajectory) {
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;

    auto phe_echos = Array<cfloat, 2>(nreadouts, nvoxels);

    // TODO: use ctx.submit_kernel or phase_encoding<<<>>>
    // dim3 block_dim = {32, 4};
    // dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    return phe_echos;
}

} // compas
