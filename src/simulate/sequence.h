#include "kmm/cuda/device.hpp"
#include "parameters/tissue.h"
#include "sequences/fisp.h"
#include "sequences/pssfp.h"

namespace compas {

Array<cfloat, 2> simulate_magnetization_pssfp(
    const CudaContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence);

Array<cfloat, 2> simulate_magnetization_fisp(
    const CudaContext& context,
    TissueParameters parameters,
    FISPSequence sequence);

}  // namespace compas