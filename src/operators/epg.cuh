#pragma once

#include "../core/complex_type.h"
#include "../core/macros.h"
#include "../core/vector.h"
#include "../core/view.h"
#include "../parameters/tissue_kernels.cuh"

namespace compas {

template<int max_N, int warp_size>
struct EPGCudaState {
    static constexpr int items_per_thread = (max_N + warp_size - 1) / warp_size;

    struct EPGStateColumn {
        cfloat F_plus;
        cfloat F_min;
        cfloat Z;
    };

    COMPAS_DEVICE
    EPGCudaState(int N) : N(N) {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i] = {0, 0, 1};
        }
    }

    COMPAS_DEVICE
    void invert(float B1) {
        float theta = static_cast<float>(M_PI * B1);
        float cos_theta = __cosf(theta);

#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].Z *= cos_theta;
        }
    }

    COMPAS_DEVICE
    void invert() {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].Z *= -1.0f;
        }
    }

    COMPAS_DEVICE
    void decay(float E1, float E2) {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].F_plus *= E2;
            state[i].F_min *= E2;
            state[i].Z *= E1;
        }
    }

    COMPAS_DEVICE
    void rotate(cfloat r) {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].F_plus *= r;
            state[i].F_min *= r.conj();
        }
    }

    COMPAS_DEVICE
    void rotate_decay(float E1, float E2, cfloat r) {
        rotate(r);
        decay(E1, E2);
    }

    COMPAS_DEVICE
    void regrowth(float E1) {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].Z += 1 - E1;
        }
    }

    COMPAS_DEVICE
    void excite(cfloat RF, float B1 = 1.0f) {
        float deg2rad = (float(M_PI) / 180.0f);

        // angle of RF pulse, convert from degrees to radians
        float alpha = deg2rad * abs(RF);
        alpha *= B1;

        // phase of RF pulse
        float phi = arg(RF);

        float x = alpha * 0.5f;
        float sinx, cosx;
        __sincosf(x, &sinx, &cosx);
        float sinx_sq = sinx * sinx;
        float cosx_sq = cosx * cosx;

        // double angle formula
        float sinalpha = 2.0f * sinx * cosx;
        float cosalpha = 2.0f * cosx_sq - 1.0f;

        // phase stuff
        float sinphi, cosphi;
        __sincosf(phi, &sinphi, &cosphi);

        // again double angle formula
        float sin2phi = 2 * sinphi * cosphi;
        float cos2phi = 2 * cosphi * cosphi - 1.0f;

        // complex exponentials
        //        cfloat cfloat(cosphi,  sinphi)  = cfloat(cosphi,  sinphi);
        //        cfloat cfloat(cos2phi, sin2phi) = cfloat(cos2phi, sin2phi);
        //        cfloat conj(cfloat(cosphi,  sinphi))  = conj(cfloat(cosphi,  sinphi));
        //        cfloat conj(cfloat(cos2phi, sin2phi)) = conj(cfloat(cos2phi, sin2phi));

        // compute individual components of rotation matrix
        float R11 = cosx_sq;
        cfloat R12 = cfloat(cos2phi, sin2phi) * sinx_sq;
        cfloat R13 = -cfloat(0, 1) * cfloat(cosphi, sinphi) * sinalpha;

        cfloat R21 = cfloat(cos2phi, -sin2phi) * sinx_sq;
        float R22 = cosx_sq;
        cfloat R23 = cfloat(cosphi, -sinphi) * sinalpha;

        cfloat R31 = -cfloat(0, 1) * cfloat(cosphi, -sinphi) * (sinalpha * 0.5f);
        cfloat R32 = cfloat(0, 1) * cfloat(cosphi, sinphi) * (sinalpha * 0.5f);
        float R33 = cosalpha;

        // apply rotation matrix to each state
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            auto a = state[i];
            EPGStateColumn b;

            b.F_plus = R11 * a.F_plus + R12 * a.F_min + R13 * a.Z;
            b.F_min = R21 * a.F_plus + R22 * a.F_min + R23 * a.Z;
            b.Z = R31 * a.F_plus + R32 * a.F_min + R33 * a.Z;

            state[i] = b;
        }
    }

    COMPAS_DEVICE
    void excite(float RF, float B1 = 1.0f) {
        // TODO: optimize for when RF is not complex
        excite(cfloat(RF), B1);
    }

    COMPAS_DEVICE
    void sample_transverse(cfloat* output, index_t index) const {
        if (is_index_local(index)) {
            index_t i = index_to_local_item(index);
            *output += state[i].F_plus;
        }
    }

    COMPAS_DEVICE
    void shift_down() {
        unsigned int mask = ~0u;
        int src_lane = (my_lane() + 1) % warp_size;

#pragma unroll items_per_thread
        for (index_t i = 0; i < items_per_thread; i++) {
            auto input = state[i].F_min;

            if (my_lane() == 0 && i + 1 < items_per_thread) {
                input = state[i + 1].F_min;
            }

            state[i].F_min = __shfl_sync(mask, input, src_lane, warp_size);
        }

        if (is_index_local(N - 1)) {
            index_t i = index_to_local_item(N - 1);
            state[i].F_min = 0;
        }
    }

    COMPAS_DEVICE
    void shift_up() {
        unsigned int mask = ~0u;
        int src_lane = (my_lane() + warp_size - 1) % warp_size;

#pragma unroll items_per_thread
        for (index_t i = items_per_thread - 1; i >= 0; i--) {
            auto input = state[i].F_plus;

            if (my_lane() + 1 == warp_size && i > 0) {
                input = state[i - 1].F_plus;
            }

            state[i].F_plus = __shfl_sync(mask, input, src_lane, warp_size);
        }

        if (is_index_local(0)) {
            index_t i = index_to_local_item(0);
            state[i].F_plus = conj(state[i].F_min);
        }
    }

    COMPAS_DEVICE
    void dephasing() {
        shift_down();
        shift_up();
    }

    COMPAS_DEVICE
    static index_t my_lane() {
        return threadIdx.x % warp_size;
    }

    COMPAS_DEVICE
    static index_t is_index_local(index_t i) {
        return i % warp_size == my_lane();
    }

    COMPAS_DEVICE
    static index_t index_to_local_item(index_t i) {
        return i / warp_size;
    }

  private:
    int N;
    EPGStateColumn state[items_per_thread];
};

}  // namespace compas