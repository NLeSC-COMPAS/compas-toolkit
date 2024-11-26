#include "compas/core/utils.h"
#include "compas/simulate/pssfp.h"
#include "pssfp_kernels.cuh"

namespace compas {

template<int batch_size>
int simulate_pssfp_sequence_batch(
    const kmm::DeviceContext& context,
    int iz,
    const cuda_view_mut<cfloat, 2>& echos,
    const TissueParametersView& parameters,
    const pSSFPSequenceView& sequence) {
    int nreadouts = sequence.nTR;
    int nvoxels = parameters.nvoxels;
    int nz = kmm::checked_cast<int>(sequence.z.size());

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    dim3 block_size = 256;
    dim3 grid_size = div_ceil(uint(nvoxels * batch_size), block_size.x);

    while (iz + batch_size <= nz) {
        auto z_subslices = cuda_view<float> {sequence.z.data() + iz, {{batch_size}}};

        kernels::simulate_pssfp<batch_size><<<grid_size, block_size, 0, context.stream()>>>(  //
            echos,
            z_subslices,
            parameters,
            sequence);

        iz += batch_size;
    }

    return iz;
}

void simulate_magnetization_kernel(
    const kmm::DeviceContext& context,
    kmm::NDRange,
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

Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence) {
    int nreadouts = kmm::checked_cast<int>(sequence.RF_train.size());
    int nvoxels = parameters.nvoxels;
    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    void (*fun)(
        const kmm::DeviceContext&,
        kmm::NDRange,
        cuda_view_mut<cfloat, 2>,
        TissueParametersView,
        pSSFPSequenceView) = simulate_magnetization_kernel;

    context.submit_device(nvoxels, fun, write(echos), parameters, sequence);
    return echos;
}
}  // namespace compas