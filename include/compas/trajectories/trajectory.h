#pragma once

#include "compas/core/object.h"

namespace compas {

/**
 * Generic class to represent a trajectory object.
 */
struct Trajectory: public Object {
    int nreadouts;
    int samples_per_readout;
    float delta_t;

    Trajectory(int nreadouts, int samples_per_readout, float delta_t) :
        nreadouts(nreadouts),
        samples_per_readout(samples_per_readout),
        delta_t(delta_t) {}
};

}  // namespace compas