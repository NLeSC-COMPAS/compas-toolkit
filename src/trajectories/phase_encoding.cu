#include "compas/trajectories/phase_encoding.h"
#include "phase_encoding_kernels.cuh"

namespace compas {

Array<cfloat, 2> phase_encoding(
    const CompasContext& ctx,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const CartesianTrajectory& trajectory) {
    using namespace kmm::placeholders;

    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;
    int chunk_size = parameters.chunk_size;

    auto ph_en_echos = Array<cfloat, 2> {{nreadouts, nvoxels}};
    dim3 block_dim = {32, 4};

    ctx.parallel_kernel(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        block_dim,
        kernels::phase_encoding,
        _xy,
        write(ph_en_echos(_y, _x)),
        echos(_y, _x),
        parameters.data(_y, _x),
        trajectory);

    return ph_en_echos;
}

}  // namespace compas
