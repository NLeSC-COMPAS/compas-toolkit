#pragma once

#include "core/complex_type.h"
#include "core/context.h"
#include "fisp_kernels.cuh"

namespace compas {

struct FISPSequence {
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

    FISPSequenceView view() const {
        return {
            .RF_train = RF_train.view(),
            .sliceprofiles = sliceprofiles.view(),
            .TR = TR,
            .TE = TE,
            .max_state = max_state,
            .TI = TI};
    }
};

FISPSequence make_fisp_sequence(
    const CudaContext& context,
    host_view<cfloat> RF_train,
    host_view<cfloat, 2> sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI);

}  // namespace compas