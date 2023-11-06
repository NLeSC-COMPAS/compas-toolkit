#pragma once

#include "macros.h"

namespace compas {

template<typename T>
COMPAS_HOST_DEVICE constexpr T div_ceil(T a, T b) {
    return (a / b) + (a % b == 0 ? T(0) : T(1));
}

template<typename T>
COMPAS_HOST_DEVICE constexpr bool is_power_of_two(const T& value) {
    return value > 0 && (((value - T(1)) & value) == 0);
}

template<typename T>
COMPAS_HOST_DEVICE T round_up_to_multiple_of(T n, T k) {
    T remainder = n % k;
    return n + (remainder > 0 ? k - remainder : 0);
}

}  // namespace compas