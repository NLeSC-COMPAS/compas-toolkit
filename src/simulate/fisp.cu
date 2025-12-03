#include "compas/core/utils.h"
#include "compas/simulate/fisp.h"
#include "fisp_kernels.cuh"

namespace compas {

template<int max_N, int warp_size = max_N>
void simulate_fisp_sequence_for_size(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUSubviewMut<cfloat, 2> transposed_echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    static constexpr int threads_per_block = 256;
    static constexpr int warps_per_block = threads_per_block / warp_size;

    int RF_train_length = int(sequence.RF_train.size());
    int nvoxels = voxels.size();

    COMPAS_CHECK(is_power_of_two(warp_size) && warp_size <= 32);
    COMPAS_CHECK(sequence.max_state <= max_N);
    COMPAS_CHECK(sequence.sliceprofiles.size(1) == sequence.RF_train.size());
    COMPAS_CHECK(transposed_echos.begin(0) == voxels.begin);
    COMPAS_CHECK(transposed_echos.end(0) == voxels.end);
    COMPAS_CHECK(transposed_echos.size(1) == RF_train_length);

    context.fill(transposed_echos.data(), transposed_echos.size(), cfloat(0));

    dim3 block_size = {warp_size, warps_per_block};
    dim3 grid_size = {1, uint(div_ceil(nvoxels, warps_per_block))};

    for (index_t i = 0; i < sequence.sliceprofiles.size(0); i++) {
        kernels::simulate_fisp<max_N, warp_size, warps_per_block>
            <<<grid_size, block_size, 0, context>>>(
                voxels,
                transposed_echos,
                sequence.sliceprofiles.drop_axis(i),
                parameters,
                sequence);

        KMM_GPU_CHECK(gpuGetLastError());
    }
}

static void simulate_magnetization_kernel(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUSubviewMut<cfloat, 2> transposed_echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    if (sequence.max_state <= 4) {
        simulate_fisp_sequence_for_size<4, 2>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 8) {
        simulate_fisp_sequence_for_size<8, 4>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 16) {
        simulate_fisp_sequence_for_size<16, 8>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 32) {
        simulate_fisp_sequence_for_size<32, 16>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 64) {
        simulate_fisp_sequence_for_size<64, 32>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 96) {
        simulate_fisp_sequence_for_size<96, 32>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else if (sequence.max_state <= 128) {
        simulate_fisp_sequence_for_size<128, 32>(
            context,
            voxels,
            transposed_echos,
            parameters,
            sequence);
    } else {
        COMPAS_ERROR("max_state cannot exceed 128");
    }
}

Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    FISPSequence sequence) {
    using namespace kmm::placeholders;
    auto RF_length = int(sequence.RF_train.size());
    auto nvoxels = parameters.nvoxels;
    auto chunk_size = parameters.chunk_size;
    auto transposed_echos = Array<cfloat, 2> {{nvoxels, RF_length}};

    context.parallel_device(
        nvoxels,
        chunk_size,
        simulate_magnetization_kernel,
        _x,
        write(transposed_echos(_x, _)),
        read(parameters, _x),
        sequence);

    auto undersampling_factor = sequence.undersampling_factor;
    auto nreadouts = RF_length * undersampling_factor;
    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    context.parallel_kernel(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        {32, 32},
        kernels::expand_undersampled_echos,
        _x,
        _y,
        undersampling_factor,
        write(echos[_][_x]),
        transposed_echos[_x][_]);

    return echos;
}

}  // namespace compas
