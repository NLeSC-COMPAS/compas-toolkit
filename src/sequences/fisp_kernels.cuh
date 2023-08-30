#pragma once

#include "core/view.h"

namespace compas {
struct FISPSequenceView {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    cuda_view<cfloat> RF_train;

    // Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    // Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
    cuda_view<cfloat, 2> sliceprofiles;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Echo time in seconds, assumed constant during the sequence
    float TE;

    // Maximum number of states to keep track of in EPG simulation
    int max_state;

    // Inversion delay after the inversion prepulse in seconds
    float TI;
};
}  // namespace compas