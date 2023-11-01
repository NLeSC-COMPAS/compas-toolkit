#pragma once

#include "../operators/epg.cuh"
#include "../parameters/tissue_view.cuh"
#include "fisp3d_view.h"

namespace compas {
namespace kernels {
template<int max_N, int warp_size>
COMPAS_DEVICE void simulate_fisp3d_for_voxel(
    const FISP3DSequenceView& sequence,
    cuda_strided_view_mut<cfloat> echos,
    TissueVoxel p) {
    auto off_resonance_rotation = [](float delta_t, float B0 = 0.0f) -> cfloat {
        if (B0 == 0.0f) {
            return 1;
        }

        auto theta = float(2.0f * M_PI) * delta_t * B0;
        return polar(1.0f, theta);
    };

    auto calculate_E = [](float delta_t, float T) -> float {
        return __expf(-delta_t * (1.0f / T));
    };

    auto T1 = p.T1;
    auto T2 = p.T2;
    auto TR = sequence.TR;
    auto TE = sequence.TE;
    auto TI = sequence.TI;
    auto TW = sequence.TW;

    auto E1_TE = calculate_E(TE, T1);
    auto E2_TE = calculate_E(TE, T2);

    auto E1_TI = calculate_E(TI, T1);
    auto E2_TI = calculate_E(TI, T2);

    auto E1_TR_minus_TE = calculate_E(TR - TE, T1);
    auto E2_TR_minus_TE = calculate_E(TR - TE, T2);

    auto E1_TW = calculate_E(TW, T1);
    auto E2_TW = calculate_E(TW, T2);

    auto r_TE = off_resonance_rotation(TE, p.B0);
    auto r_TR_minus_TE = off_resonance_rotation(TR - TE, p.B0);
    auto r_TW = off_resonance_rotation(TW, p.B0);

    auto omega = EPGThreadBlockState<max_N, warp_size>(sequence.max_state);

    // repeat sequence 4 times to reach hyper steady state
    for (index_t j = 0; j < 4; j++) {

        // apply inversion pulse
        omega.invert();
        omega.spoil();
        omega.decay(E1_TI, E2_TI);
        omega.regrowth(E1_TI);

        for (index_t i = 0; i < sequence.RF_train.size(); i++) {
            // mix states
            omega.excite(sequence.RF_train[i], p.B1);
            // T2 decay F states, T1 decay Z states, B0 rotation until TE
            omega.rotate_decay(E1_TE, E2_TE, r_TE);
            omega.regrowth(E1_TE);
            // don't sample F₊[0]
            // T2 decay F states, T1 decay Z states, B0 rotation until next RF excitation
            omega.rotate_decay(E1_TR_minus_TE, E2_TR_minus_TE, r_TR_minus_TE);
            omega.regrowth(E1_TR_minus_TE);
            // shift F states due to dephasing gradients
            omega.dephasing();
        }

        omega.spoil();
        omega.rotate_decay(E1_TW, E2_TW, r_TW);
        omega.regrowth(E1_TW);
    }

    // apply inversion pulse
    omega.invert();
    omega.spoil();
    omega.decay(E1_TI, E2_TI);
    omega.regrowth(E1_TI);

    for (index_t i = 0; i < sequence.RF_train.size(); i++) {

        // mix states
        omega.excite(sequence.RF_train[i], p.B1);
        // T2 decay F states, T1 decay Z states, B0 rotation until TE
        omega.rotate_decay(E1_TE, E2_TE, r_TE);
        omega.regrowth(E1_TE);
        // sample F₊[0]
        omega.sample_transverse(&echos[2*i], 0);
        omega.sample_transverse(&echos[2*i+1], 0);
        // T2 decay F states, T1 decay Z states, B0 rotation until next RF excitation
        omega.rotate_decay(E1_TR_minus_TE, E2_TR_minus_TE, r_TR_minus_TE);
        omega.regrowth(E1_TR_minus_TE);
        // shift F states due to dephasing gradients
        omega.dephasing();
    }
}
}  // namespace kernels
}  // namespace compas