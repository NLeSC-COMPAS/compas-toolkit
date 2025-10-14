#include "compas/core/context.h"
#include "compas/core/view.h"
#include "kernel_float.h"

namespace compas {

enum struct GemmComputeMethod {
    Pedantic,  // highest precision
    Regular,  // balanced
    Fast  // highest performance
};

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
    GemmComputeMethod kind = GemmComputeMethod::Regular);

void compute_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 2> result,
    GPUSubview<kernel_float::bfloat16_t, 2> lhs,
    GPUSubview<kernel_float::bfloat16_t, 2> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind = GemmComputeMethod::Regular);

void compute_complex_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> result,
    GPUSubview<float, 3> lhs,
    GPUSubview<float, 3> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind = GemmComputeMethod::Regular);

void compute_complex_gemm(
    const kmm::DeviceResource& context,
    GPUSubviewMut<float, 3> result,
    GPUSubview<kernel_float::bfloat16_t, 3> lhs,
    GPUSubview<kernel_float::bfloat16_t, 3> rhs,
    float alpha,
    float beta,
    GemmComputeMethod kind = GemmComputeMethod::Regular);

void convert_complex_to_planar(
    const kmm::DeviceResource& context,  //
    GPUSubviewMut<float, 3> output,
    GPUSubview<cfloat, 2> input);

void convert_complex_to_planar(
    const kmm::DeviceResource& context,  //
    GPUSubviewMut<kernel_float::bfloat16_t, 3> output,
    GPUSubview<cfloat, 2> input);

void convert_planar_to_complex(
    const kmm::DeviceResource& context,  //
    GPUSubviewMut<cfloat, 2> output,
    GPUSubview<float, 3> input);

}  // namespace compas