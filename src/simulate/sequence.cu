#include "core/utils.h"
#include "sequence.h"
#include "sequence_kernels.cuh"

namespace compas {

template<int batch_size>
int simulate_sequence_batch(
    int iz,
    const CudaArray<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const pSSFPSequence& sequence) {
    int nreadouts = sequence.nTR;
    int nvoxels = parameters.nvoxels;
    int nz = sequence.z.size();

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    dim3 block_size = 256;
    dim3 grid_size = div_ceil(uint(nvoxels * batch_size), block_size.x);

    while (iz + batch_size <= nz) {
        kernels::simulate_pssfp<batch_size><<<grid_size, block_size>>>(
            echos.view_mut(),
            sequence.z.slice(iz, iz + batch_size).view(),
            parameters.view(),
            sequence.view());

        iz += batch_size;
    }

    return iz;
}

void simulate_sequence(
    const CudaContext& context,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence) {
    // Initialize echos to zero
    echos.fill(0);

    // We process the z-slices in batches. The first batch processes 32 slices at once, the next 16 slices, the next
    // 8 slices, etc. Since the reduction is performed using warp-shuffles, these batch sizes must powers of two and
    // cannot exceed 32.  The offset keeps track of how many slices have already been processed and is incremented
    // by each call to `simulate_sequence_batch`.
    int offset = 0;
    offset = simulate_sequence_batch<32>(offset, echos, parameters, sequence);
    offset = simulate_sequence_batch<16>(offset, echos, parameters, sequence);
    offset = simulate_sequence_batch<8>(offset, echos, parameters, sequence);
    offset = simulate_sequence_batch<4>(offset, echos, parameters, sequence);
    offset = simulate_sequence_batch<2>(offset, echos, parameters, sequence);
    simulate_sequence_batch<1>(offset, echos, parameters, sequence);
}

}  // namespace compas