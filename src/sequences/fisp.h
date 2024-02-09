#pragma once

#include "core/assertion.h"
#include "core/complex_type.h"
#include "core/context.h"
#include "fisp_view.h"

namespace compas {

struct FISPSequence: public Object {
    // Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    // angle.(RF_train) should be the RF phases in degrees.
    CudaArray<cfloat> RF_train;

    // Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    // Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
    CudaArray<cfloat, 2> sliceprofiles;

    // Repetition time in seconds, assumed constant during the sequence
    float TR;

    // Echo time in seconds, assumed constant during the sequence
    float TE;

    // Maximum number of states to keep track of in EPG simulation
    int max_state;

    // Inversion delay after the inversion prepulse in seconds
    float TI;

    FISPSequence(
        CudaArray<cfloat> RF_train,
        CudaArray<cfloat, 2> sliceprofiles,
        float TR,
        float TE,
        int max_state,
        float TI) :
        RF_train(RF_train),
        sliceprofiles(sliceprofiles),
        TR(TR),
        TE(TE),
        max_state(max_state),
        TI(TI) {}
};

inline FISPSequence make_fisp_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    host_view<cfloat, 2> sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    COMPAS_ASSERT(sliceprofiles.size(1) == RF_train.size(0));
    return {context.allocate(RF_train), context.allocate(sliceprofiles), TR, TE, max_state, TI};
}

}  // namespace compas

namespace kmm {

template<>
struct TaskArgument<ExecutionSpace::Cuda, compas::FISPSequence> {
    using type = compas::FISPSequenceView;

    static TaskArgument pack(TaskBuilder& builder, compas::FISPSequence p) {
        return {
            {.RF_train = {},
             .sliceprofiles = {},
             .TR = p.TR,
             .TE = p.TE,
             .max_state = p.max_state,
             .TI = p.TI},
            pack_argument<ExecutionSpace::Cuda>(builder, p.RF_train),
            pack_argument<ExecutionSpace::Cuda>(builder, p.sliceprofiles)};
    }

    type unpack(TaskContext& context) {
        view.RF_train = unpack_argument<ExecutionSpace::Cuda>(context, RF_train);
        view.sliceprofiles = unpack_argument<ExecutionSpace::Cuda>(context, sliceprofiles);
        return view;
    }

    type view;
    PackedArray<const compas::cfloat> RF_train;
    PackedArray<const compas::cfloat, 2> sliceprofiles;
};

};  // namespace kmm