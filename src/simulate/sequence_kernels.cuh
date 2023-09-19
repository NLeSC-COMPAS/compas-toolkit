#include "sequences/fisp_kernels.cuh"
#include "sequences/pssfp_kernels.cuh"

namespace compas {
namespace kernels {

template<int warp_size>
__global__ void simulate_pssfp(
    cuda_view_mut<cfloat, 2> echos,
    cuda_view<float> z,
    TissueParametersView parameters,
    pSSFPSequenceView sequence) {
    index_t lane = threadIdx.x % warp_size;
    index_t voxel = index_t(blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
    index_t nvoxels = parameters.nvoxels;

    if (voxel >= nvoxels) {
        return;
    }

    simulate_pssfp_for_voxel<warp_size>(
        sequence,
        z[lane],
        echos.drop_axis<1>(voxel),
        parameters.get(voxel));
}

template<int max_N, int warp_size>
__global__ void simulate_fisp(
    cuda_view_mut<cfloat, 2> echos,
    cuda_view<cfloat> slice_profile,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    index_t voxel = index_t(blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
    index_t nvoxels = parameters.nvoxels;

    if (voxel >= nvoxels) {
        return;
    }

    simulate_fisp_for_voxel<max_N, warp_size>(
        sequence,
        slice_profile,
        echos.drop_axis<1>(voxel),
        parameters.get(voxel));
}

}  // namespace kernels
}  // namespace compas