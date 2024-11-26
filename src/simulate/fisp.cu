#include "compas/core/utils.h"
#include "compas/simulate/fisp.h"
#include "fisp_kernels.cuh"

namespace compas {

template<int max_N, int warp_size = max_N>
void simulate_fisp_sequence_for_size(
    const kmm::DeviceContext& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    COMPAS_ASSERT(sequence.max_state <= max_N);
    COMPAS_ASSERT(is_power_of_two(warp_size) && warp_size <= 32);

    int nvoxels = parameters.nvoxels;
    int nreadouts = int(sequence.RF_train.size());

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    context.fill(echos, cfloat(0));

    for (index_t i = 0; i < sequence.sliceprofiles.size(0); i++) {
        dim3 block_size = 256;
        dim3 grid_size = div_ceil(uint(nvoxels * warp_size), block_size.x);

        kernels::simulate_fisp<max_N, warp_size><<<grid_size, block_size, 0, context>>>(
            echos,
            sequence.sliceprofiles.drop_axis<0>(i),
            parameters,
            sequence);
    }
}

void simulate_magnetization_kernel(
    const kmm::DeviceContext& context,
    kmm::NDRange,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    if (sequence.max_state <= 4) {
        simulate_fisp_sequence_for_size<4, 2>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 8) {
        simulate_fisp_sequence_for_size<8, 4>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 16) {
        simulate_fisp_sequence_for_size<16, 8>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 32) {
        simulate_fisp_sequence_for_size<32, 16>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 64) {
        simulate_fisp_sequence_for_size<64, 32>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 96) {
        simulate_fisp_sequence_for_size<96, 32>(context, echos, parameters, sequence);
    } else if (sequence.max_state <= 128) {
        simulate_fisp_sequence_for_size<128, 32>(context, echos, parameters, sequence);
    } else {
        COMPAS_PANIC("max_state cannot exceed 128");
    }
}

Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    FISPSequence sequence) {
    int nreadouts = int(sequence.RF_train.size());
    int nvoxels = parameters.nvoxels;
    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    void (*fun)(
        const kmm::DeviceContext&,
        kmm::NDRange,
        cuda_view_mut<cfloat, 2>,
        TissueParametersView,
        FISPSequenceView) = simulate_magnetization_kernel;

    context.submit_device(nvoxels, fun, write(echos), parameters, sequence);
    return echos;
}

}  // namespace compas
