#pragma once

#include "core/complex_type.h"
#include "core/view.h"

namespace compas {
struct RepetitionData {
    float ex;  // Data for excitation pulse
    float inv;  // Data for inversion delay
    float pr;  // Data for preparation interval
};

struct pSSFPSequenceView {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    cuda_view<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Time-discretized RF waveform, normalized to flip angle of 1 degree
    cuda_view<cfloat> gamma_dt_RF;

    // Time intervals
    RepetitionData dt;  // Δt
    RepetitionData gamma_dt_GRz;  // γΔtGRz

    // Vector with different positions along the slice direction.
    cuda_view<float> z;
};
}  // namespace compas