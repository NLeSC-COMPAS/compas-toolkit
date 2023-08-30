#include "fisp.h"

namespace compas {

FISPSequence make_fisp_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    host_view<cfloat, 2> sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    COMPAS_ASSERT(sliceprofiles.size(1) == RF_train.size(0));

    return {
        .RF_train = context.allocate(RF_train),
        .sliceprofiles = context.allocate(sliceprofiles),
        .TR = TR,
        .TE = TE,
        .max_state = max_state,
        .TI = TI};
}
}  // namespace compas