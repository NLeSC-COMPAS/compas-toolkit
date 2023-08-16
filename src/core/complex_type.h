#pragma once

#include <cmath>

#include "macros.h"

namespace compas {

template<typename T>
struct complex_storage {
    T re = {}, im = {};
};

template<>
struct alignas(2 * sizeof(float)) complex_storage<float> {
    float re = 0, im = 0;
};

template<>
struct alignas(2 * sizeof(double)) complex_storage<double> {
    double re = 0, im = 0;
};

template<typename T>
struct alignas(2 * sizeof(T)) complex_type: complex_storage<T> {
    COMPAS_HOST_DEVICE
    complex_type(T real = {}, T imag = {}) : complex_storage<T> {real, imag} {}

    COMPAS_HOST_DEVICE
    T real() const {
        return this->re;
    }

    COMPAS_HOST_DEVICE
    T imag() const {
        return this->im;
    }

    COMPAS_HOST_DEVICE
    T norm() const {
        return real() * real() + imag() * imag();
    }

    COMPAS_HOST_DEVICE
    T conj() const {
        return {real(), -imag()};
    }
};

COMPAS_HOST_DEVICE
static complex_type<float> polar(float mag, float angle = {}) {
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
float arg(complex_type<float> m) {
    return atan2f(m.im, m.re);
}

COMPAS_HOST_DEVICE
double arg(complex_type<double> m) {
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

COMPAS_HOST_DEVICE complex_type<double> log(complex_type<double> v) {
    return {::log(abs(v)), arg(v)};
}

COMPAS_HOST_DEVICE complex_type<float> log(complex_type<float> v) {
#if COMPAS_IS_DEVICE
    return {::__logf(abs(v)), arg(v)};
#else
    return {::logf(abs(v)), arg(v)};
#endif
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
    return exp(b * ::log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(complex_type<T> a) {
    return a;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(complex_type<T> a) {
    return {-a.re, -a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(complex_type<T> a, complex_type<T> b) {
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
COMPAS_HOST_DEVICE complex_type<T> operator-(complex_type<T> a, complex_type<T> b) {
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
COMPAS_HOST_DEVICE complex_type<T> operator*(complex_type<T> a, complex_type<T> b) {
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
COMPAS_HOST_DEVICE complex_type<T> operator/(complex_type<T> x, complex_type<T> y) {
    T norm = T(1) / (y.re * y.re + y.im * y.im);

    return {(x.re * y.re + x.im * y.im) * norm, (x.im * y.re - x.re * y.im) * norm};
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
COMPAS_HOST_DEVICE complex_type<T> operator+=(complex_type<T>& a, complex_type<T> b) {
    return a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-=(complex_type<T>& a, complex_type<T> b) {
    return a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*=(complex_type<T>& a, complex_type<T> b) {
    return a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/=(complex_type<T>& a, complex_type<T> b) {
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

using cfloat = complex_type<float>;
using cdouble = complex_type<double>;

}  // namespace compas