#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/context.h"

namespace compas {

/**
 * Returns the result of `lhs - rhs`.
 *
 * Additionally, if `objective_out` is provided, this function writes the sum over the squares of the differences
 * between `lhs` and `rhs` to `objective_out`.
 *
 * @param ctx CUDA context.
 * @param lhs The left-hand side.
 * @param rhs The right-hand side.
 * @param objective_out Optionally, pointer to where the sum over the squares of the differences must be stored.
 * @return
 */
Array<cfloat, 3> compute_residual(
    CompasContext ctx,
    Array<cfloat, 3> lhs,
    Array<cfloat, 3> rhs,
    float* objective_out = nullptr);

};  // namespace compas