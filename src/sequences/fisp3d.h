#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "fisp3d_view.h"

namespace compas {

struct FISP3DSequence {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    CudaArray<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Echo time in seconds, assumed constant during the sequence
    float TE;

    // Maximum number of states to keep track of in EPG simulation
    int max_state;

    // Inversion delay after the inversion prepulse in seconds
    float TI;

    // Delay after RF cycle in seconds
    float TW;

    FISP3DSequenceView view() const {
        return {
            .RF_train = RF_train.view(),
            .TR = TR,
            .TE = TE,
            .max_state = max_state,
            .TI = TI,
            .TW = TW};
    }
};

FISP3DSequence make_fisp3d_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    float TR,
    float TE,
    int max_state,
    float TI,
    float TW);

}  // namespace compas