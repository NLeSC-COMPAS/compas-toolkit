#pragma once

#include "core/assertion.h"
#include "core/context.h"
#include "pssfp_view.h"

namespace compas {
struct pSSFPSequence: public Object {
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
        nRF(gamma_dt_RF.size()),
        nTR(RF_train.size()),
        gamma_dt_RF(gamma_dt_RF),
        dt(dt),
        gamma_dt_GRz(gamma_dt_GRz),
        nz(z.size()),
        z(z) {}
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

namespace kmm {
template<>
struct TaskArgument<ExecutionSpace::Cuda, compas::pSSFPSequence> {
    using type = compas::pSSFPSequenceView;

    static TaskArgument pack(TaskBuilder& builder, compas::pSSFPSequence p) {
        return {
            {.nTR = p.nTR,
             .RF_train = {},
             .TR = p.TR,
             .gamma_dt_RF = {},
             .dt = p.dt,
             .gamma_dt_GRz = p.gamma_dt_GRz,
             .z = {}},
            pack_argument<ExecutionSpace::Cuda>(builder, p.RF_train),
            pack_argument<ExecutionSpace::Cuda>(builder, p.gamma_dt_RF),
            pack_argument<ExecutionSpace::Cuda>(builder, p.z),
        };
    }

    type unpack(TaskContext& context) {
        view.RF_train = unpack_argument<ExecutionSpace::Cuda>(context, RF_train);
        view.gamma_dt_RF = unpack_argument<ExecutionSpace::Cuda>(context, gamma_dt_RF);
        view.z = unpack_argument<ExecutionSpace::Cuda>(context, z);
        return view;
    }

    compas::pSSFPSequenceView view;
    PackedArray<const compas::cfloat> RF_train;
    PackedArray<const compas::cfloat> gamma_dt_RF;
    PackedArray<const float> z;
};

};  // namespace kmm