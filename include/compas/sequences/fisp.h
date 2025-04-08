#pragma once

#include "compas/core/assertion.h"
#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "fisp_view.h"

namespace compas {

struct FISPSequence {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    Array<cfloat> RF_train;

    // Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    // Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
    Array<cfloat, 2> sliceprofiles;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Echo time in seconds, assumed constant during the sequence
    float TE;

    float TW = 0;

    // Maximum number of states to keep track of in EPG simulation
    int max_state;

    // Inversion delay after the inversion prepulse in seconds
    float TI;

    int undersampling_factor = 1;

    int repetitions = 1;

    // With or without inversion prepulse at the start of every repetition
    bool inversion_prepulse = true;

    // Spoiling is assumed before the start of a next cycle
    bool wait_spoiling = true;
};

inline FISPSequence make_fisp_sequence(
    const CompasContext& context,
    View<cfloat> RF_train,
    View<cfloat, 2> sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    return {
        context.allocate(RF_train),
        context.allocate(sliceprofiles),
        TR,
        TE,
        0.0F,
        max_state,
        TI};
}

}  // namespace compas

KMM_DEFINE_STRUCT_ARGUMENT(
    compas::FISPSequence,
    it.RF_train,
    it.sliceprofiles,
    it.TR,
    it.TE,
    it.TW,
    it.max_state,
    it.TI,
    it.undersampling_factor,
    it.repetitions,
    it.inversion_prepulse,
    it.wait_spoiling)

KMM_DEFINE_STRUCT_VIEW(compas::FISPSequence, compas::FISPSequenceView)