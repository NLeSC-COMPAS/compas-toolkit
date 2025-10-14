#pragma once

#include "compas/core/backends.h"
#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

namespace compas {

/**
 * Object to represent a view of a Cartesian trajectory.
 */
struct CartesianTrajectoryView {
    int nreadouts;
    int samples_per_readout;
    float delta_t;
    GPUView<cfloat> k_start;
    cfloat delta_k;

    /**
     * Apply prephaser.
     *
     * @param readout_idx
     * @param m
     * @param p
     * @return
     */
    COMPAS_DEVICE
    cfloat calculate_readout_magnetization(index_t readout_idx, cfloat m, TissueVoxel p) const {
        auto R2 = 1 / p.T2;
        auto ns = samples_per_readout;
        auto x = p.x;
        // auto y = p.y;

        // undo T2 decay and B0 phase that happened between start readout and echo time
        cfloat arg = cfloat(-R2, float(2 * M_PI) * p.B0);
        cfloat result = -0.5f * float(ns) * delta_t * arg;

        // apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
        auto k0 = -0.5f * float(ns) * delta_k;  //k_start[readout_idx];
        result += cfloat(0, k0.re * x /*+ k0.im * y*/);

        return m * exp(result);
    }

    /**
     * Apply readout gradient, T₂ decay and B₀ rotation.
     *
     * @param p
     * @return
     */
    COMPAS_DEVICE
    cfloat calculate_sample_phase_decay(int sample_index, TissueVoxel p) const {
        auto R2 = 1.0F / p.T2;
        auto x = p.x;
        auto y = p.y;

        // apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k.re * x + delta_k.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return exp(float(sample_index) * cfloat(-delta_t * R2, Theta));
    }

    COMPAS_DEVICE
    cfloat calculate_sample_decay_absolute(int sample_index, TissueVoxel p) const {
        auto R2 = 1.0F / p.T2;
        auto x = p.x;
        auto y = p.y;

        // There are ns samples per readout, echo time is assumed to occur
        // at index (ns/2)+1. Now compute sample index relative to the echo time
        float s = float(sample_index) - 0.5f * float(samples_per_readout);

        // Apply readout gradient, T₂ decay and B₀ rotation
        auto Theta = delta_k.re * x + delta_k.im * y;
        Theta += delta_t * float(2 * M_PI) * p.B0;

        return exp(s * cfloat(-delta_t * R2, Theta));
    }

    COMPAS_DEVICE
    cfloat calculate_sample_decay_absolute_delta_T2(int sample_index, TissueVoxel p) const {
        auto R2 = 1.0f / p.T2;

        // There are ns samples per readout, echo time is assumed to occur
        // at index (ns/2)+1. Now compute sample index relative to the echo time
        float s = float(sample_index) - 0.5f * float(samples_per_readout);

        auto Es = calculate_sample_decay_absolute(sample_index, p);
        return (s * delta_t * R2 * R2) * Es;
    }
};

}  // namespace compas