#include "compas/parameters/tissue.h"
#include "compas/sequences/fisp.h"
#include "compas/sequences/pssfp.h"

namespace compas {

Array<cfloat, 2> simulate_magnetization(
    const CudaContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence);

void simulate_magnetization_kernel(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    pSSFPSequenceView sequence);

Array<cfloat, 2> simulate_magnetization(
    const CudaContext& context,
    TissueParameters parameters,
    FISPSequence sequence);

void simulate_magnetization_kernel(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence);

}  // namespace compas