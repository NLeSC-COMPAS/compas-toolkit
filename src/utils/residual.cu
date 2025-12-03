#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/utils/residual.h"
#include "residual_kernels.cuh"

namespace compas {

Array<cfloat, 3> compute_residual(
    CompasContext ctx,
    Array<cfloat, 3> lhs,
    Array<cfloat, 3> rhs,
    float* objective_out) {
    COMPAS_CHECK(lhs.size() == rhs.size());
    auto n = kmm::checked_cast<int>(lhs.size().volume());
    auto d = lhs.size();

    static int constexpr block_dim = 256;
    int num_blocks = std::min(1024, div_ceil(n, block_dim));

    auto output = Array<cfloat, 3> {d};
    auto objective = Array<float> {1};
    auto partials = Array<float> {num_blocks};

    ctx.submit_kernel(
        uint(num_blocks),
        uint(block_dim),
        kernels::calculate_elementwise_difference<block_dim>,
        n,
        lhs,
        rhs,
        write(output),
        write(partials));

    ctx.submit_kernel(
        1,
        block_dim,
        kernels::accumulate_partial_sums<block_dim>,
        partials,
        write(objective));

    if (objective_out != nullptr) {
        objective.copy_to(objective_out);
    }

    return output;
}

}  // namespace compas