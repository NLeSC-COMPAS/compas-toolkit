#pragma once

namespace compas {

template<typename T>
T div_ceil(T a, T b) {
    return (a / b) + (a % b == 0 ? T(0) : T(1));
}

}  // namespace compas