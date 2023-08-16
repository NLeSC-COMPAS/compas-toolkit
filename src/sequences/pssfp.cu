#include "pssfp.h"

namespace compas {

pSSFPSequence make_pssfp_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    float TR,
    host_view<cfloat> gamma_dt_RF,
    RepetitionData dt,
    RepetitionData gamma_dt_GRz,
    host_view<float> z) {
    COMPAS_ASSERT(RF_train.size() > 0);

    return {
        .nTR = RF_train.size(),
        .RF_train = context.allocate(RF_train),
        .TR = TR,
        .nRF = gamma_dt_RF.size(),
        .gamma_dt_RF = context.allocate(gamma_dt_RF),
        .dt = dt,
        .gamma_dt_GRz = gamma_dt_GRz,
        .nz = z.size(),
        .z = context.allocate(z),
    };
}
}  // namespace compas