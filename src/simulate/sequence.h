#include "kmm/cuda/device.hpp"
#include "parameters/tissue.h"
#include "sequences/fisp.h"
#include "sequences/pssfp.h"

namespace compas {

Array<cfloat, 2> simulate_magnetization(
    const CudaContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence);

Array<cfloat, 2> simulate_magnetization(
    const CudaContext& context,
    TissueParameters parameters,
    FISPSequence sequence);

void simulate_magnetization_kernel(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    pSSFPSequenceView sequence);

void simulate_magnetization_kernel(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence);

}  // namespace compas