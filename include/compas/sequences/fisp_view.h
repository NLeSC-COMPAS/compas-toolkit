#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/view.h"

namespace compas {
struct FISPSequenceView {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    GPUView<cfloat> RF_train;

    // Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    // Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
    GPUView<cfloat, 2> sliceprofiles;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Echo time in seconds, assumed constant during the sequence
    float TE;

    float TW = 0;

    // Maximum number of states to keep track of in EPG simulation
    int max_state;

    // Inversion delay after the inversion prepulse in seconds
    float TI;

    int undersampling_factor;

    int repetitions;

    bool inversion_prepulse;

    bool wait_spoiling;
};
}  // namespace compas