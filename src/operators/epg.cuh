#pragma once

#include <cfloat>

#include "compas/core/complex_type.h"
#include "compas/core/macros.h"
#include "compas/core/vector.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

struct EPGStateColumn {
    cfloat F_plus = 0;
    cfloat F_min = 0;
    cfloat Z = 0;
};

struct EPGExciteMatrix {
    float R11;
    float R12_re;
    float R12_im;
    float R13_re;
    float R13_im;
    //    cfloat R21;
    //    float R22;
    //    cfloat R23;
    //    cfloat R31;
    //    cfloat R32;
    float R33;
    float unused;

    COMPAS_DEVICE
    EPGExciteMatrix() = default;

    COMPAS_DEVICE
    EPGExciteMatrix(cfloat RF, float B1 = 1.0f) {
        static constexpr float MIN_ANGLE = FLT_MIN;
        static constexpr float DEG2RAD = (float(M_PI) / 180.0f);

        // angle of RF pulse
        float abs_RF = hypotf(RF.re, RF.im);

        // convert from degrees to radians
        float alpha = DEG2RAD * abs_RF * B1;

        float x = alpha * 0.5f;
        float sinx, cosx;
        __sincosf(x, &sinx, &cosx);
        float sinx_sq = sinx * sinx;
        float cosx_sq = cosx * cosx;

        // double angle formula
        float sinalpha = 2.0f * sinx * cosx;
        float cosalpha = 2.0f * cosx_sq - 1.0f;

        // phase of RF pulse
        float cosphi = 1;
        float sinphi = 0;
        //        float phi = arg(RF);
        //        __sincosf(phi, &sinphi, &cosphi);

        // normalizing RF is faster than using sincosf
        if (abs_RF > MIN_ANGLE) {
            cosphi = RF.re * rhypotf(RF.re, RF.im);
            sinphi = RF.im * rhypotf(RF.re, RF.im);
        }

        // again double angle formula
        float sin2phi = (2.0f * cosphi) * sinphi;
        float cos2phi = (2.0f * cosphi) * cosphi - 1.0f;

        // compute individual components of rotation matrix
        this->R11 = cosx_sq;
        //        this->R12 = cfloat(cos2phi * sinx_sq, sin2phi * sinx_sq);
        //        this->R13 = cfloat(sinphi * sinalpha, -cosphi * sinalpha);

        this->R12_re = cos2phi * sinx_sq;
        this->R12_im = sin2phi * sinx_sq;
        this->R13_re = sinphi * sinalpha;
        this->R13_im = -cosphi * sinalpha;

        // this->R21 = cfloat(cos2phi * sinx_sq, -sin2phi * sinx_sq);
        // this->R22 = cfloat(cosx_sq, 0.0f);
        // this->R23 = cfloat(sinphi * sinalpha, cosphi * sinalpha);

        // this->R31 = cfloat(-0.5f * sinphi * sinalpha, -0.5f * cosphi * sinalpha);
        // this->R32 = cfloat(-0.5f * sinphi * sinalpha, 0.5f * cosphi * sinalpha);
        this->R33 = cosalpha;
    }

    COMPAS_DEVICE
    EPGStateColumn apply(EPGStateColumn x) {
        // This values are stored
        auto R12 = cfloat(R12_re, R12_im);
        auto R13 = cfloat(R13_re, R13_im);

        // These values can be derived from the other entries
        auto R21 = conj(R12);
        auto R22 = R11;
        auto R23 = conj(R13);

        auto R31 = cfloat(-0.5F * R13.re, 0.5F * R13.im);
        auto R32 = conj(R31);

        EPGStateColumn result;
        result.F_plus = (R11 * x.F_plus).add_mul(R12, x.F_min).add_mul(R13, x.Z);
        result.F_min = (R22 * x.F_min).add_mul(R21, x.F_plus).add_mul(R23, x.Z);
        result.Z = (R31 * x.F_plus).add_mul(R32, x.F_min).add_mul(R33, x.Z);
        return result;
    }
};

template<int max_N, int warp_size, int warps_per_block>
struct EPGThreadBlockState {
    static constexpr int items_per_thread = (max_N + warp_size - 1) / warp_size;

    COMPAS_DEVICE
    EPGThreadBlockState(int N) : N(N) {
        initialize();
    }

    COMPAS_DEVICE
    void initialize() {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i] = {0, 0, 0};

            if (local_to_global_index(i) == 0) {
                state[i].Z = 1;
            }
        }
    }

    COMPAS_DEVICE
    void invert(float B1) {
        float theta = static_cast<float>(M_PI) * B1;
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
    void spoil() {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].F_plus = 0.0F;
            state[i].F_min = 0.0F;
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
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i].F_plus *= E2 * r;
            state[i].F_min *= (E2 * r).conj();
            state[i].Z *= E1;
        }
    }

    COMPAS_DEVICE
    void regrowth(float E1) {
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            if (local_to_global_index(i) == 0) {
                state[i].Z += 1 - E1;
            }
        }
    }

    COMPAS_DEVICE
    void excite(cfloat RF, float B1) {
        excite(EPGExciteMatrix(RF, B1));
    }

    COMPAS_DEVICE
    void excite(EPGExciteMatrix m) {
        // apply rotation matrix to each state
#pragma unroll items_per_thread
        for (int i = 0; i < items_per_thread; i++) {
            state[i] = m.apply(state[i]);
        }
    }

    COMPAS_DEVICE
    bool sample_transverse(index_t index, cfloat* output) const {
#pragma unroll items_per_thread
        for (index_t i = 0; i < items_per_thread; i++) {
            if (global_to_local_item(index) == i) {
                *output = state[i].F_plus;
                return global_to_local_lane(index) == my_lane();
            }
        }

        return false;
    }

    COMPAS_DEVICE
    void dephasing() {
        shift_down();
        shift_up();
    }

  private:
    COMPAS_DEVICE
    void shift_down() {
#ifdef COMPAS_IS_CUDA
        unsigned int mask = 0xFFFFFFFF;
#elif defined(COMPAS_IS_HIP)
        long long unsigned int mask = 0xFFFFFFFFFFFFFFFF;
#endif

        auto old_first = state[0].F_min;
        auto new_last = old_first;
#if defined(COMPAS_IS_CUDA)
        new_last = __shfl_down_sync(mask, old_first, 1, warp_size);
#elif defined(COMPAS_IS_HIP)
        new_last.re = __shfl_down_sync(mask, old_first.re, 1, warp_size);
        new_last.im = __shfl_down_sync(mask, old_first.im, 1, warp_size);
#endif

        // Shift all F_min values one up
#pragma unroll
        for (int i = 0; i < items_per_thread - 1; i++) {
            state[i].F_min = state[i + 1].F_min;
        }

        state[items_per_thread - 1].F_min = new_last;

#pragma unroll items_per_thread
        for (index_t i = 0; i < items_per_thread; i++) {
            if (local_to_global_index(i) >= N - 1) {
                state[i].F_min = 0;
            }
        }
    }

    COMPAS_DEVICE
    void shift_up() {
#ifdef COMPAS_IS_CUDA
        unsigned int mask = 0xFFFFFFFF;
#elif defined(COMPAS_IS_HIP)
        long long unsigned int mask = 0xFFFFFFFFFFFFFFFF;
#endif
        int src_lane = (my_lane() + (warp_size - 1)) % warp_size;

        auto old_last = state[items_per_thread - 1].F_plus;
        auto new_first = old_last;
#if defined(COMPAS_IS_CUDA)
        new_first = __shfl_up_sync(mask, old_last, 1, warp_size);
#elif defined(COMPAS_IS_HIP)
        new_first.re = __shfl_up_sync(mask, old_last.re, 1, warp_size);
        new_first.im = __shfl_up_sync(mask, old_last.im, 1, warp_size);
#endif

        // Shift all F_plus values one down
#pragma unroll
        for (int i = 0; i < items_per_thread - 1; i++) {
            state[items_per_thread - 1 - i].F_plus = state[items_per_thread - 2 - i].F_plus;
        }

        state[0].F_plus = new_first;

        if (my_lane() == 0) {
            state[0].F_plus = conj(state[0].F_min);
        }
    }

    COMPAS_DEVICE
    static index_t my_lane() {
        return threadIdx.x % warp_size;
    }

    COMPAS_DEVICE
    static index_t local_to_global_index(index_t i) {
        return my_lane() * items_per_thread + i;
    }

    COMPAS_DEVICE
    static index_t global_to_local_lane(index_t i) {
        return i / items_per_thread;
    }

    COMPAS_DEVICE
    static index_t global_to_local_item(index_t i) {
        return i % items_per_thread;
    }

    int N;
    EPGStateColumn state[items_per_thread];
};

}  // namespace compas