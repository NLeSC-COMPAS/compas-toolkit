#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {
namespace kernels {
__global__ void add_difference_to_parameters(
    int nvoxels,
    cuda_view_mut<float, 2> new_parameters,
    cuda_view<float, 2> old_parameters,
    int target_field,
    float delta) {
    auto v = index_t(blockIdx.x * blockDim.x + threadIdx.x);

    if (v < nvoxels) {
        for (auto field = 0; field < TissueParameterField::NUM_FIELDS; field++) {
            new_parameters[field][v] = old_parameters[field][v];
        }

        new_parameters[target_field][v] += delta;
    }
}

__global__ void calculate_finite_difference(
    int nreadouts,
    int nvoxels,
    cuda_view_mut<cfloat, 2> delta_echos,
    cuda_view<cfloat, 2> echos,
    float inv_delta) {
    auto v = index_t(blockIdx.x * blockDim.x + threadIdx.x);
    auto r = index_t(blockIdx.y * blockDim.y + threadIdx.y);

    if (r < nreadouts && v < nvoxels) {
        delta_echos[r][v] = (delta_echos[r][v] - echos[r][v]) * inv_delta;
    }
}
}  // namespace kernels
}  // namespace compas
