#pragma once

#include <cmath>

#include "macros.h"

namespace compas {

template<typename T>
struct alignas(2 * sizeof(T)) complex_type {
    COMPAS_HOST_DEVICE
    complex_type(T real = {}, T imag = {}) : re(real), im(imag) {}

    COMPAS_HOST_DEVICE
    T real() const {
        return re;
    }

    COMPAS_HOST_DEVICE
    T imag() const {
        return im;
    }

    COMPAS_HOST_DEVICE
    T norm() const {
        return re * re + im * im;
    }

    COMPAS_HOST_DEVICE
    T conj() const {
        return {re, -im};
    }

    T re;
    T im;
};

using cfloat = complex_type<float>;
using cdouble = complex_type<double>;

COMPAS_HOST_DEVICE
complex_type<float> polar(float mag, float angle = {}) {
    float sin, cos;
#if COMPAS_IS_DEVICE
    __sincosf(angle, &sin, &cos);
#else
    sin = ::sinf(angle);
    cos = ::cosf(angle);
#endif
    return {mag * cos, mag * sin};
}

COMPAS_HOST_DEVICE
complex_type<double> polar(double mag, double angle = {}) {
    return {mag * ::cos(angle), mag * ::sin(angle)};
}

COMPAS_HOST_DEVICE
complex_type<float> exp(complex_type<float> m) {
#if COMPAS_IS_DEVICE
    float e = ::__expf(m.re);
#else
    float e = ::expf(m.re);
#endif
    return polar(e, m.im);
}

COMPAS_HOST_DEVICE
complex_type<double> exp(complex_type<double> m) {
    return polar(::exp(m.re), m.im);
}

COMPAS_HOST_DEVICE
complex_type<float> arg(complex_type<float> m) {
    return atan2f(m.im, m.re);
}

COMPAS_HOST_DEVICE
complex_type<double> arg(complex_type<double> m) {
    return atan2(m.im, m.re);
}

COMPAS_HOST_DEVICE
double abs(complex_type<double> a) {
    return hypot(a.re, a.im);
}

COMPAS_HOST_DEVICE
float abs(complex_type<float> a) {
    return hypotf(a.re, a.im);
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> log(complex_type<T> v) {
    return {log(abs(v)), arg(v)};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(complex_type<T> a, complex_type<T> b) {
    return exp(b * log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(complex_type<T> a, T b) {
    return exp(b * log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(T a, complex_type<T> b) {
    return exp(b * log(a));
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
operator/(complex_type<T> x, complex_type<T> y) {
    T norm = T(1) / (y.re * y.re + y.im * y.im);

    return {
        (x.re * y.re + x.im * y.im) * norm,
        (x.im * y.re - x.re * y.im) * norm};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/(complex_type<T> x, T y) {
    return x * T(T(1) / y);
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/(T x, complex_type<T> y) {
    T norm = T(1) / (y.re * y.re + y.im * y.im);

    return {(x * y.re) * norm, (x * y.im) * norm};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator+=(complex_type<T>& a, complex_type<T> b) {
    return a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator-=(complex_type<T>& a, complex_type<T> b) {
    return a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator*=(complex_type<T>& a, complex_type<T> b) {
    return a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
operator/=(complex_type<T>& a, complex_type<T> b) {
    return a = a / b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+=(complex_type<T>& a, T b) {
    return a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-=(complex_type<T>& a, T b) {
    return a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*=(complex_type<T>& a, T b) {
    return a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/=(complex_type<T>& a, T b) {
    return a = a / b;
}

}  // namespace compas