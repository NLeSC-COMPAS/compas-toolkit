#pragma once

#include "core/context.h"
#include "pssfp_view.h"

namespace compas {
struct pSSFPSequence: public Object {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    CudaArray<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // nr of RF discretization points
    int nRF;

    // number of repetition times
    int nTR;

    // Time-discretized RF waveform, normalized to flip angle of 1 degree
    CudaArray<cfloat> gamma_dt_RF;

    // Time intervals
    RepetitionData dt;  // Δt
    RepetitionData gamma_dt_GRz;  // γΔtGRz

    // Number of spins in the z-direction
    int nz;

    // Vector with different positions along the slice direction.
    CudaArray<float> z;

    pSSFPSequence(
        CudaArray<cfloat> RF_train,
        float TR,
        CudaArray<cfloat> gamma_dt_RF,
        RepetitionData dt,
        RepetitionData gamma_dt_GRz,
        CudaArray<float> z) :
        RF_train(RF_train),
        TR(TR),
        nRF(gamma_dt_RF.size()),
        nTR(RF_train.size()),
        gamma_dt_RF(gamma_dt_RF),
        dt(dt),
        gamma_dt_GRz(gamma_dt_GRz),
        nz(z.size()),
        z(z) {}

    pSSFPSequenceView view() const {
        return {
            .nTR = nTR,
            .RF_train = RF_train.view(),
            .TR = TR,
            .gamma_dt_RF = gamma_dt_RF.view(),
            .dt = dt,
            .gamma_dt_GRz = gamma_dt_GRz,
            .z = z.view()};
    }
};

inline pSSFPSequence make_pssfp_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    float TR,
    host_view<cfloat> gamma_dt_RF,
    RepetitionData dt,
    RepetitionData gamma_dt_GRz,
    host_view<float> z) {
    COMPAS_ASSERT(RF_train.size() > 0);

    return {
        context.allocate(RF_train),
        TR,
        context.allocate(gamma_dt_RF),
        dt,
        gamma_dt_GRz,
        context.allocate(z),
    };
}

}  // namespace compas