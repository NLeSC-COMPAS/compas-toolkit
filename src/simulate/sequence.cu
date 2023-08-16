#include "core/utils.h"
#include "sequence.h"
#include "sequence_kernels.cuh"

namespace compas {

void simulate_sequence(
    const CudaContext& context,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence) {
    int nreadouts = sequence.nTR;
    int nvoxels = parameters.nvoxels;

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    echos.fill(0);

    dim3 block_size = 256;
    dim3 grid_size = div_ceil(uint(nvoxels), block_size.x);
    kernels::simulate_pssfp<<<grid_size, block_size>>>(
        echos.view_mut(),
        parameters.view(),
        sequence.view());
}
}  // namespace compas