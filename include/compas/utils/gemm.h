#include "compas/core/context.h"
#include "compas/core/view.h"

namespace kernel_float {
    using bfloat16_t = uint16_t;
}

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
        GPUSubviewMut<float, 2> result,
        GPUSubview<float, 2> lhs,
        GPUSubview<float, 2> rhs,
        float alpha,
        float beta,
        GemmComputeMethod kind = GemmComputeMethod::Fast);

void compute_gemm(
        const kmm::DeviceResource& context,
        GPUSubviewMut<float, 2> result,
        GPUSubview<kernel_float::bfloat16_t, 2> lhs,
        GPUSubview<kernel_float::bfloat16_t, 2> rhs,
        float alpha,
        float beta,
        GemmComputeMethod kind = GemmComputeMethod::Fast);

void compute_gemm(
        const kmm::DeviceResource& context,
        GPUSubviewMut<float, 2> result_re,
        GPUSubviewMut<float, 2> result_im,
        GPUSubview<float, 2> lhs_re,
        GPUSubview<float, 2> lhs_im,
        GPUSubview<float, 2> rhs_re,
        GPUSubview<float, 2> rhs_im,
        float alpha,
        float beta,
        GemmComputeMethod kind = GemmComputeMethod::Fast);

void compute_gemm(
        const kmm::DeviceResource& context,
        GPUSubviewMut<float, 2> result_re,
        GPUSubviewMut<float, 2> result_im,
        GPUSubview<kernel_float::bfloat16_t, 2> lhs_re,
        GPUSubview<kernel_float::bfloat16_t, 2> lhs_im,
        GPUSubview<kernel_float::bfloat16_t, 2> rhs_re,
        GPUSubview<kernel_float::bfloat16_t, 2> rhs_im,
        float alpha,
        float beta,
        GemmComputeMethod kind = GemmComputeMethod::Fast);

}  // namespace compas