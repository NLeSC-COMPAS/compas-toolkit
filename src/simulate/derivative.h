#include "kmm/cuda/device.hpp"
#include "parameters/tissue.h"
#include "sequences/fisp.h"
#include "sequences/pssfp.h"

namespace compas {

static constexpr float DEFAULT_FINITE_DIFFERENCE_DELTA = 1e-4F;

Array<cfloat, 2> simulate_magnetization_derivative(
    const CudaContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence,
    float delta = DEFAULT_FINITE_DIFFERENCE_DELTA);

Array<cfloat, 2> simulate_magnetization_derivative(
    const CudaContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    FISPSequence sequence,
    float delta = DEFAULT_FINITE_DIFFERENCE_DELTA);

}  // namespace compas