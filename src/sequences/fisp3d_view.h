#pragma once

#include "core/complex_type.h"
#include "core/view.h"

namespace compas {
struct FISP3DSequenceView {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    cuda_view<cfloat> RF_train;

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
};
}  // namespace compas