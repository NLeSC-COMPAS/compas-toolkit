#pragma once

#include "compas/core/assertion.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "pssfp_view.h"

namespace compas {
struct pSSFPSequence {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    Array<cfloat> RF_train;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // nr of RF discretization points
    int nRF;

    // number of repetition times
    int nTR;

    // Time-discretized RF waveform, normalized to flip angle of 1 degree
    Array<cfloat> gamma_dt_RF;

    // Time intervals
    RepetitionData dt;  // Δt
    RepetitionData gamma_dt_GRz;  // γΔtGRz

    // Number of spins in the z-direction
    int nz;

    // Vector with different positions along the slice direction.
    Array<float> z;

    pSSFPSequence(
        Array<cfloat> RF_train,
        float TR,
        Array<cfloat> gamma_dt_RF,
        RepetitionData dt,
        RepetitionData gamma_dt_GRz,
        Array<float> z) :
        RF_train(RF_train),
        TR(TR),
        nRF(kmm::checked_cast<int>(gamma_dt_RF.size())),
        nTR(kmm::checked_cast<int>(RF_train.size())),
        gamma_dt_RF(gamma_dt_RF),
        dt(dt),
        gamma_dt_GRz(gamma_dt_GRz),
        nz(kmm::checked_cast<int>(z.size())),
        z(z) {}
};

inline pSSFPSequence make_pssfp_sequence(
    const CompasContext& context,
    View<cfloat> RF_train,
    float TR,
    View<cfloat> gamma_dt_RF,
    RepetitionData dt,
    RepetitionData gamma_dt_GRz,
    View<float> z) {
    COMPAS_CHECK(RF_train.size() > 0);

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

KMM_DEFINE_STRUCT_ARGUMENT(
    compas::pSSFPSequence,
    it.nTR,
    it.RF_train,
    it.TR,
    it.gamma_dt_RF,
    it.dt,
    it.gamma_dt_GRz,
    it.z)

KMM_DEFINE_STRUCT_VIEW(compas::pSSFPSequence, compas::pSSFPSequenceView)