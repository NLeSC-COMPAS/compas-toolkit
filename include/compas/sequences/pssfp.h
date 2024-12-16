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

namespace kmm {
template<>
struct Argument<compas::pSSFPSequence> {
    using type = compas::pSSFPSequenceView;

    static Argument pack(TaskBuilder& builder, compas::pSSFPSequence p) {
        return {
            {.nTR = p.nTR,
             .RF_train = {},
             .TR = p.TR,
             .gamma_dt_RF = {},
             .dt = p.dt,
             .gamma_dt_GRz = p.gamma_dt_GRz,
             .z = {}},
            pack_argument(builder, p.RF_train),
            pack_argument(builder, p.gamma_dt_RF),
            pack_argument(builder, p.z),
        };
    }

    template<ExecutionSpace space>
    type unpack(TaskContext& context) {
        view.RF_train = unpack_argument<space>(context, RF_train);
        view.gamma_dt_RF = unpack_argument<space>(context, gamma_dt_RF);
        view.z = unpack_argument<space>(context, z);
        return view;
    }

    compas::pSSFPSequenceView view;
    packed_argument_t<Array<compas::cfloat>> RF_train;
    packed_argument_t<Array<compas::cfloat>> gamma_dt_RF;
    packed_argument_t<Array<float>> z;
};

};  // namespace kmm