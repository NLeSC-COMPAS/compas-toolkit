#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {
namespace kernels {
__global__ void add_difference_to_parameters(
    kmm::Range<index_t> voxels,
    GPUSubviewMut<float, 2> new_parameters,
    GPUSubview<float, 2> old_parameters,
    int target_field,
    float delta) {
    auto v = index_t(blockIdx.x * blockDim.x + threadIdx.x) + voxels.begin;

    if (v < voxels.end) {
        for (auto field = 0; field < TissueParameterField::NUM_FIELDS; field++) {
            new_parameters[field][v] = old_parameters[field][v];
        }

        new_parameters[target_field][v] += delta;
    }
}

__global__ void calculate_finite_difference(
    kmm::Bounds<2, index_t> range,
    GPUSubviewMut<cfloat, 2> delta_echos,
    GPUSubview<cfloat, 2> echos0,
    GPUSubview<cfloat, 2> echos1,
    float inv_delta) {
    auto v = index_t(blockIdx.x * blockDim.x + threadIdx.x) + range.x.begin;
    auto r = index_t(blockIdx.y * blockDim.y + threadIdx.y) + range.y.begin;

    if (range.contains(v, r)) {
        delta_echos[r][v] = (echos1[r][v] - echos0[r][v]) * inv_delta;
    }
}
}  // namespace kernels
}  // namespace compas
