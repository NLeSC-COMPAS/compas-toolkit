#include "fisp3d.h"

namespace compas {

FISP3DSequence make_fisp3d_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    float TR,
    float TE,
    int max_state,
    float TI,
    float TW) {

    return {
        .RF_train = context.allocate(RF_train),
        .TR = TR,
        .TE = TE,
        .max_state = max_state,
        .TI = TI,
        .TW = TW};
}
}  // namespace compas