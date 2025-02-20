#pragma once

#include "compas/core/complex_type.h"
#include "compas/core/view.h"

namespace compas {
struct RepetitionData {
    float ex;  // Data for excitation pulse
    float inv;  // Data for inversion delay
    float pr;  // Data for preparation interval
};

struct pSSFPSequenceView {
    // number of repetition times
    int nTR;

    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    GPUView<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Time-discretized RF waveform, normalized to flip angle of 1 degree
    GPUView<cfloat> gamma_dt_RF;

    // Time intervals
    RepetitionData dt;  // Δt
    RepetitionData gamma_dt_GRz;  // γΔtGRz

    // Vector with different positions along the slice direction.
    GPUView<float> z;
};
}  // namespace compas