#pragma once

#include "macros.h"

namespace compas {

template<typename T>
COMPAS_HOST_DEVICE constexpr T div_ceil(T a, T b) {
    return (a / b) + (a % b == 0 ? T(0) : T(1));
}

template<typename T>
COMPAS_HOST_DEVICE constexpr bool is_power_of_two(const T& value) {
    if (!(value > T(0))) {
        return false;
    }

    for (int i = 0; i < 64; i++) {
        auto v = uint64_t(1) << uint64_t(i);
        if (T(v) == value) {
            return true;
        }
    }

    return false;
}

}  // namespace compas