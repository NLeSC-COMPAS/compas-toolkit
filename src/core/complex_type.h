#pragma once
#include "defines.h"

namespace compas {

template<typename T>
struct alignas(2 * sizeof(T)) complex_type {
    COMPAS_HOST_DEVICE
    complex_type(T r = {}, T i = {}) : re(r), im(i) {}

    T real() const {
        return re;
    }

    T imag() const {
        return im;
    }

    T re;
    T im;
};

using cfloat = complex_type<float>;
using cdouble = complex_type<double>;

COMPAS_HOST_DEVICE
complex_type<float> exp(complex_type<float> m) {
    float sin, cos, e;
#if COMPAS_IS_DEVICE
    __sincosf(m.im, &sin, &cos);
    e = ::__expf(m.re);
#else
    sin = ::sinf(m.im);
    cos = ::cosf(m.im);
    e = ::expf(m.re);
#endif
    return {e * cos, e * sin};
}

COMPAS_HOST_DEVICE
complex_type<double> exp(complex_type<double> m) {
    double e = ::exp(m.re);
    double a = m.im;
    return {e * ::cos(a), e * ::sin(a)};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator+(complex_type<T> a, complex_type<T> b) {
    return {a.re + b.re, a.im + b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(complex_type<T> a, T b) {
    return {a.re + b, a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(T a, complex_type<T> b) {
    return {a + b.re, b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator-(complex_type<T> a, complex_type<T> b) {
    return {a.re - b.re, a.im - b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(complex_type<T> a, T b) {
    return {a.re - b, a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(T a, complex_type<T> b) {
    return {a - b.re, b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator*(complex_type<T> a, complex_type<T> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*(complex_type<T> a, T b) {
    return {a.re * b, a.im * b};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*(T a, complex_type<T> b) {
    return {a * b.re, a * b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator+=(complex_type<T>& a, complex_type<T> b) {
    a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator-=(complex_type<T>& a, complex_type<T> b) {
    a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator*=(complex_type<T>& a, complex_type<T> b) {
    a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator/=(complex_type<T>& a, complex_type<T> b) {
    a = a / b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+=(complex_type<T>& a, T b) {
    a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-=(complex_type<T>& a, T b) {
    a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*=(complex_type<T>& a, T b) {
    a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/=(complex_type<T>& a, T b) {
    a = a / b;
}

}  // namespace compas