#include "parameters/tissue.h"
#include "sequences/fisp.h"
#include "sequences/fisp3d.h"
#include "sequences/pssfp.h"

namespace compas {

void simulate_sequence(
    const CudaContext& context,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence);

void simulate_sequence(
    const CudaContext& context,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    FISPSequence sequence);

void simulate_sequence(
    const CudaContext& context,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    FISP3DSequence sequence);

}  // namespace compas