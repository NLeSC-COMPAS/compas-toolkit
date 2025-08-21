#include "compas/core/utils.h"
#include "compas/simulate/pssfp.h"
#include "pssfp_kernels.cuh"

namespace compas {

template<int batch_size>
int simulate_pssfp_sequence_batch(
    const kmm::DeviceResource& context,
    int nvoxels,
    int iz,
    const GPUSubviewMut<cfloat, 2>& echos,
    const TissueParametersView& parameters,
    const pSSFPSequenceView& sequence) {
    int nreadouts = sequence.nTR;
    int nz = kmm::checked_cast<int>(sequence.z.size());

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.size(1) == nvoxels);

    dim3 block_size = 256;
    dim3 grid_size = div_ceil(uint(nvoxels * batch_size), block_size.x);

    while (iz + batch_size <= nz) {
        auto z_subslices = GPUView<float> {sequence.z.data() + iz, {{batch_size}}};

        kernels::simulate_pssfp<batch_size><<<grid_size, block_size, 0, context.stream()>>>(  //
            nvoxels,
            echos,
            z_subslices,
            parameters,
            sequence);

        iz += batch_size;
    }

    return iz;
}

void simulate_magnetization_kernel(
    const kmm::DeviceResource& context,
    kmm::Range<index_t>,
    int nvoxels,
    GPUSubviewMut<cfloat, 2> echos,
    TissueParametersView parameters,
    pSSFPSequenceView sequence) {
    // Initialize echos to zero
    context.fill(echos.data(), echos.size(), cfloat(0));

    // We process the z-slices in batches. The first batches process 32 slices at once, the next batches process 16
    // slices at once, the next 8 slices, etc. Since the reduction is performed using warp-shuffles, these batch sizes
    // must powers of two and cannot exceed 32.  The offset keeps track of how many slices have already been processed
    // and is incremented by each call to `simulate_pssfp_sequence_batch`.
    int offset = 0;
    offset =
        simulate_pssfp_sequence_batch<32>(context, nvoxels, offset, echos, parameters, sequence);
    offset =
        simulate_pssfp_sequence_batch<16>(context, nvoxels, offset, echos, parameters, sequence);
    offset =
        simulate_pssfp_sequence_batch<8>(context, nvoxels, offset, echos, parameters, sequence);
    offset =
        simulate_pssfp_sequence_batch<4>(context, nvoxels, offset, echos, parameters, sequence);
    offset =
        simulate_pssfp_sequence_batch<2>(context, nvoxels, offset, echos, parameters, sequence);
    offset =
        simulate_pssfp_sequence_batch<1>(context, nvoxels, offset, echos, parameters, sequence);

    COMPAS_CHECK(offset == sequence.z.size());
}

Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence) {
    using namespace kmm::placeholders;

    int nreadouts = kmm::checked_cast<int>(sequence.RF_train.size(0));
    int nvoxels = parameters.nvoxels;
    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    void (*fun)(
        const kmm::DeviceResource&,
        kmm::Range<index_t>,
        int,
        GPUSubviewMut<cfloat, 2>,
        TissueParametersView,
        pSSFPSequenceView) = simulate_magnetization_kernel;

    context.submit_device(fun, _x, nvoxels, write(echos), parameters, sequence);
    return echos;
}
}  // namespace compas
