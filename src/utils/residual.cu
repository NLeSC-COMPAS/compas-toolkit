#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/utils/residual.h"
#include "residual_kernels.cuh"

namespace compas {

Array<cfloat, 3> compute_residual(
    CudaContext ctx,
    Array<cfloat, 3> lhs,
    Array<cfloat, 3> rhs,
    float* objective_out) {
    COMPAS_ASSERT(lhs.sizes() == rhs.sizes());
    auto n = lhs.size();
    auto d = lhs.sizes();

    static int constexpr block_dim = 256;
    int num_blocks = std::min(1024, div_ceil(n, block_dim));

    auto output = Array<cfloat> {n};
    auto objective = Array<float> {1};
    auto partials = Array<float> {num_blocks};

    ctx.submit_kernel(
        uint(num_blocks),
        uint(block_dim),
        kernels::calculate_elementwise_difference<block_dim>,
        lhs.flatten(),
        rhs.flatten(),
        write(output),
        write(partials));

    if (objective_out != nullptr) {
        ctx.submit_kernel(
            1,
            block_dim,
            kernels::accumulate_partial_sums<block_dim>,
            partials,
            write(objective));

        *objective_out = objective.read()[0];
    }

    return output.reshape(d[0], d[1], d[2]);
}

}  // namespace compas