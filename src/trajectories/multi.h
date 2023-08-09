#pragma once

#include <variant>

#include "cartesian.h"
#include "spiral.h"

namespace compas {

struct Trajectory {
    Trajectory(const SpiralTrajectory& s) : inner_(s) {}
    Trajectory(const CartesianTrajectory& s) : inner_(s) {}

    const SpiralTrajectory* as_spiral() const {
        return std::get_if<SpiralTrajectory>(&inner_);
    }

    const CartesianTrajectory* as_cartesian() const {
        return std::get_if<CartesianTrajectory>(&inner_);
    }

  private:
    std::variant<SpiralTrajectory, CartesianTrajectory> inner_;
};

}  // namespace compas