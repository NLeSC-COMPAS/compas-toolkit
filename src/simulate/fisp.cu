#include "compas/core/utils.h"
#include "compas/simulate/fisp.h"
#include "fisp_kernels.cuh"

namespace compas {

template<int max_N, int warp_size = max_N>
void simulate_fisp_sequence_for_size(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUSubviewMut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    int nreadouts = int(sequence.RF_train.size()) * sequence.undersampling_factor;

    COMPAS_CHECK(is_power_of_two(warp_size) && warp_size <= 32);
    COMPAS_CHECK(sequence.max_state <= max_N);
    COMPAS_CHECK(sequence.sliceprofiles.size(1) == sequence.RF_train.size());
    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.begin(1) == voxels.begin);
    COMPAS_CHECK(echos.end(1) == voxels.end);

    context.fill(echos.data(), echos.size(), cfloat(0));

    for (index_t i = 0; i < sequence.sliceprofiles.size(0); i++) {
        dim3 block_size = 256;
        dim3 grid_size = div_ceil(uint(voxels.size() * warp_size), block_size.x);

        kernels::simulate_fisp<max_N, warp_size><<<grid_size, block_size, 0, context>>>(
            voxels,
            echos,
            sequence.sliceprofiles.drop_axis(i),
            parameters,
            sequence);

        KMM_GPU_CHECK(gpuGetLastError());
    }
}

void simulate_magnetization_kernel(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUSubviewMut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    if (sequence.max_state <= 4) {
        simulate_fisp_sequence_for_size<4, 2>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 8) {
        simulate_fisp_sequence_for_size<8, 4>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 16) {
        simulate_fisp_sequence_for_size<16, 8>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 32) {
        simulate_fisp_sequence_for_size<32, 16>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 64) {
        simulate_fisp_sequence_for_size<64, 32>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 96) {
        simulate_fisp_sequence_for_size<96, 32>(context, voxels, echos, parameters, sequence);
    } else if (sequence.max_state <= 128) {
        simulate_fisp_sequence_for_size<128, 32>(context, voxels, echos, parameters, sequence);
    } else {
        COMPAS_ERROR("max_state cannot exceed 128");
    }
}

Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    FISPSequence sequence) {
    using namespace kmm::placeholders;
    int nreadouts = int(sequence.RF_train.size()) * sequence.undersampling_factor;
    int nvoxels = parameters.nvoxels;
    int chunk_size = parameters.chunk_size;
    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    void (*fun)(
        const kmm::DeviceResource&,
        kmm::Range<index_t>,
        GPUSubviewMut<cfloat, 2>,
        TissueParametersView,
        FISPSequenceView) = simulate_magnetization_kernel;

    context.parallel_device(
        nvoxels,
        chunk_size,
        fun,
        _x,
        write(echos(_, _x)),
        read(parameters, _x),
        sequence);

    return echos;
}

}  // namespace compas
