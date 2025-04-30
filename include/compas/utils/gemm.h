#include "compas/core/context.h"
#include "compas/core/view.h"

namespace compas {

enum struct GemmComputeMethod { Pedantic, Fast, BF16, TF32 };

/**
 * Multiply `lhs` and `transpose(rhs)` and write the result to `result`.
 *
 * It should be case that:
 *  * `lhs.size(0) == result.size(0)`
 *  * `rhs.size(0) == result.size(1)`
 *  * `lhs.size(1) == rhs.size(1)`
 */
void compute_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<cfloat, 2> result,
    GPUSubview<cfloat, 2> lhs,
    GPUSubview<cfloat, 2> rhs,
    cfloat beta,
    GemmComputeMethod kind = GemmComputeMethod::Fast);

}  // namespace compas