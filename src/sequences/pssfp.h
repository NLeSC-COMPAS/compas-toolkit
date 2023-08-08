#pragma once

#include "core/context.h"
#include "pssfp_kernels.cuh"

namespace compas {
struct pSSFPSequence {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    CudaArray<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Time-discretized RF waveform, normalized to flip angle of 1 degree
    CudaArray<cfloat> gamma_dt_RF;

    // Time intervals
    TimeIntervals dt;  // Δt
    TimeIntervals gamma_dt_GRz;  // γΔtGRz

    // Vector with different positions along the slice direction.
    CudaArray<float> z;

    pSSFPSequenceView view() const {
        return {
            .RF_train = RF_train.view(),
            .TR = TR,
            .gamma_dt_RF = gamma_dt_RF.view(),
            .dt = dt,
            .gamma_dt_GRz = gamma_dt_GRz,
            .z = z.view()};
    }
};
}  // namespace compas