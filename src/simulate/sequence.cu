#include "core/utils.h"
#include "sequence.h"
#include "sequences/fisp_kernels.cuh"
#include "sequences/pssfp_kernels.cuh"

namespace compas {

template<int batch_size>
int simulate_pssfp_sequence_batch(
    const kmm::CudaDevice& context,
    int iz,
    const cuda_view_mut<cfloat, 2>& echos,
    const TissueParametersView& parameters,
    const pSSFPSequenceView& sequence) {
    int nreadouts = sequence.nTR;
    int nvoxels = parameters.nvoxels;
    int nz = sequence.z.size();

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    dim3 block_size = 256;
    dim3 grid_size = div_ceil(uint(nvoxels * batch_size), block_size.x);

    while (iz + batch_size <= nz) {
        auto z_subslices = cuda_view<float> {sequence.z.data() + iz, {batch_size}};

        kernels::simulate_pssfp<batch_size><<<grid_size, block_size, 0, context.stream()>>>(  //
            echos,
            z_subslices,
            parameters,
            sequence);

        iz += batch_size;
    }

    return iz;
}

void simulate_magnetization_pssfp(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    pSSFPSequenceView sequence) {
    // Initialize echos to zero
    context.fill(echos, cfloat(0));

    // We process the z-slices in batches. The first batches process 32 slices at once, the next batches process 16
    // slices at once, the next 8 slices, etc. Since the reduction is performed using warp-shuffles, these batch sizes
    // must powers of two and cannot exceed 32.  The offset keeps track of how many slices have already been processed
    // and is incremented by each call to `simulate_pssfp_sequence_batch`.
    int offset = 0;
    offset = simulate_pssfp_sequence_batch<32>(context, offset, echos, parameters, sequence);
    offset = simulate_pssfp_sequence_batch<16>(context, offset, echos, parameters, sequence);
    offset = simulate_pssfp_sequence_batch<8>(context, offset, echos, parameters, sequence);
    offset = simulate_pssfp_sequence_batch<4>(context, offset, echos, parameters, sequence);
    offset = simulate_pssfp_sequence_batch<2>(context, offset, echos, parameters, sequence);
    offset = simulate_pssfp_sequence_batch<1>(context, offset, echos, parameters, sequence);

    COMPAS_ASSERT(offset == sequence.z.size());
}

template<int max_N, int warp_size = max_N>
void simulate_fisp_sequence_for_size(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence) {
    COMPAS_ASSERT(sequence.max_state <= max_N);
    COMPAS_ASSERT(is_power_of_two(warp_size) && warp_size <= 32);

    int nvoxels = parameters.nvoxels;
    int nreadouts = sequence.RF_train.size();

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    context.fill(echos, cfloat(0));

    for (index_t i = 0; i < sequence.sliceprofiles.size(0); i++) {
        dim3 block_size = 256;
        dim3 grid_size = div_ceil(uint(nvoxels * warp_size), block_size.x);

        kernels::simulate_fisp<max_N, warp_size><<<grid_size, block_size>>>(
            echos,
            sequence.sliceprofiles.drop_axis<0>(i),
            parameters,
            sequence);
    }
}

void simulate_magnetization_fisp(
    const kmm::CudaDevice& context,
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

}  // namespace compas
