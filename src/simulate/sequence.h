#include "kmm/cuda/device.hpp"
#include "parameters/tissue.h"
#include "sequences/fisp.h"
#include "sequences/pssfp.h"

namespace compas {

void simulate_magnetization_pssfp(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    pSSFPSequenceView sequence);

void simulate_magnetization_fisp(
    const kmm::CudaDevice& context,
    cuda_view_mut<cfloat, 2> echos,
    TissueParametersView parameters,
    FISPSequenceView sequence);

}  // namespace compas