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
    float re, im;
};

template<>
struct alignas(2 * sizeof(double)) complex_storage<double> {
    double re, im;
};

template<typename T>
struct alignas(2 * sizeof(T)) complex_type: complex_storage<T> {
    COMPAS_HOST_DEVICE
    complex_type(T real = {}, T imag = {}) {
        this->re = real;
        this->im = imag;
    }

    //    COMPAS_HOST_DEVICE
    complex_type(const complex_type<T>& that) = default;

    COMPAS_HOST_DEVICE
    complex_type(const complex_storage<T>& that) {
        this->re = that.re;
        this->im = that.im;
    }

    template<typename R>
    COMPAS_HOST_DEVICE explicit complex_type(const complex_type<R>& that) :
        complex_type {T(that.re), T(that.im)} {}

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
    complex_type<T> conj() const {
        return {real(), -imag()};
    }

    /**
     * Returns `this + a * b` or, equivalently, `fma(a, b, *this)`
     */
    COMPAS_HOST_DEVICE
    complex_type<T> add_mul(const complex_type<T>& a, const complex_type<T>& b) const {
        return fma(a, b, *this);
    }
};

template<typename T>
COMPAS_HOST_DEVICE static T real(const complex_type<T>& v) {
    return v.re;
}

template<typename T>
COMPAS_HOST_DEVICE static T imag(const complex_type<T>& v) {
    return v.im;
}

template<typename T>
COMPAS_HOST_DEVICE static complex_type<T> conj(const complex_type<T>& v) {
    return v.conj();
}

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
complex_type<float> exp(const complex_type<float>& m) {
#if COMPAS_IS_DEVICE
    float e = ::__expf(m.re);
#else
    float e = ::expf(m.re);
#endif
    return polar(e, m.im);
}

COMPAS_HOST_DEVICE
complex_type<double> exp(const complex_type<double>& m) {
    return polar(::exp(m.re), m.im);
}

COMPAS_HOST_DEVICE
float arg(const complex_type<float>& m) {
    return atan2f(m.im, m.re);
}

COMPAS_HOST_DEVICE
double arg(const complex_type<double>& m) {
    return atan2(m.im, m.re);
}

COMPAS_HOST_DEVICE
double abs(const complex_type<double>& a) {
    return hypot(a.re, a.im);
}

COMPAS_HOST_DEVICE
float abs(const complex_type<float>& a) {
    return hypotf(a.re, a.im);
}

COMPAS_HOST_DEVICE complex_type<double> log(const complex_type<double>& v) {
    return {::log(abs(v)), arg(v)};
}

COMPAS_HOST_DEVICE complex_type<float> log(const complex_type<float>& v) {
#if COMPAS_IS_DEVICE
    return {::__logf(abs(v)), arg(v)};
#else
    return {::logf(abs(v)), arg(v)};
#endif
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(const complex_type<T>& a, const complex_type<T>& b) {
    return exp(b * log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(const complex_type<T>& a, const T& b) {
    return exp(b * log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> pow(const T& a, const complex_type<T>& b) {
    return exp(b * ::log(a));
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>
fma(const complex_type<T>& a, const complex_type<T>& b, const complex_type<T>& c) {
    return {c.re + a.re * b.re - a.im * b.im, c.im + a.re * b.im + a.im * b.re};
}

template<>
COMPAS_HOST_DEVICE complex_type<float>
fma(const complex_type<float>& a, const complex_type<float>& b, const complex_type<float>& c) {
    return {fmaf(a.re, b.re, fmaf(-a.im, b.im, c.re)), fmaf(a.re, b.im, fmaf(a.im, b.re, c.im))};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(const complex_type<T>& a) {
    return a;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(const complex_type<T>& a) {
    return {-a.re, -a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(const complex_type<T>& a, const complex_type<T>& b) {
    return {a.re + b.re, a.im + b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(const complex_type<T>& a, const T& b) {
    return {a.re + b, a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator+(const T& a, const complex_type<T>& b) {
    return {a + b.re, b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(const complex_type<T>& a, const complex_type<T>& b) {
    return {a.re - b.re, a.im - b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(const complex_type<T>& a, const T& b) {
    return {a.re - b, a.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator-(const T& a, const complex_type<T>& b) {
    return {a - b.re, -b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*(const complex_type<T>& a, const complex_type<T>& b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*(const complex_type<T>& a, const T& b) {
    return {a.re * b, a.im * b};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator*(const T& a, const complex_type<T>& b) {
    return {a * b.re, a * b.im};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/(const complex_type<T>& x, const complex_type<T>& y) {
    T norm = T(1) / (y.re * y.re + y.im * y.im);

    return {(x.re * y.re + x.im * y.im) * norm, (x.im * y.re - x.re * y.im) * norm};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/(const complex_type<T>& x, const T& y) {
    return x * T(T(1) / y);
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T> operator/(const T& x, const complex_type<T>& y) {
    T norm = T(x) / (y.re * y.re + y.im * y.im);

    return {y.re * norm, -y.im * norm};
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator+=(complex_type<T>& a, const complex_type<T>& b) {
    return a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator-=(complex_type<T>& a, const complex_type<T>& b) {
    return a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator*=(complex_type<T>& a, const complex_type<T>& b) {
    return a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator/=(complex_type<T>& a, const complex_type<T>& b) {
    return a = a / b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator+=(complex_type<T>& a, const T& b) {
    return a = a + b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator-=(complex_type<T>& a, const T& b) {
    return a = a - b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator*=(complex_type<T>& a, const T& b) {
    return a = a * b;
}

template<typename T>
COMPAS_HOST_DEVICE complex_type<T>& operator/=(complex_type<T>& a, const T& b) {
    return a = a / b;
}

template<typename T>
COMPAS_HOST_DEVICE bool operator==(const complex_type<T>& a, const complex_type<T>& b) {
    return a.real() == b.real() && a.imag() == b.imag();
}

template<typename T>
COMPAS_HOST_DEVICE bool operator==(const complex_type<T>& a, const T& b) {
    return operator==(a, complex_type<T>(b));
}

template<typename T>
COMPAS_HOST_DEVICE bool operator==(const T& a, const complex_type<T>& b) {
    return operator==(complex_type<T>(a), b);
}

template<typename T>
COMPAS_HOST_DEVICE bool operator!=(const complex_type<T>& a, const complex_type<T>& b) {
    return !operator==(a, b);
}

template<typename T>
COMPAS_HOST_DEVICE bool operator!=(const complex_type<T>& a, const T& b) {
    return !operator==(a, b);
}

template<typename T>
COMPAS_HOST_DEVICE bool operator!=(const T& a, const complex_type<T>& b) {
    return !operator==(a, b);
}

using cfloat = complex_type<float>;
using cdouble = complex_type<double>;

}  // namespace compas

#ifdef COMPAS_IS_CUDA
    #define COMPAS_COMPLEX_DEVICE_SHFL_IMPL(F, Ty)                                 \
        template<typename T>                                                       \
        COMPAS_DEVICE compas::complex_type<T> F(                                   \
            unsigned mask,                                                         \
            const compas::complex_type<T>& var,                                    \
            Ty arg,                                                                \
            int width = 32) {                                                      \
            return {::F(mask, var.re, arg, width), ::F(mask, var.im, arg, width)}; \
        }
#elif defined(COMPAS_IS_HIP)
    #define COMPAS_COMPLEX_DEVICE_SHFL_IMPL(F, Ty)                                 \
        template<typename T>                                                       \
        COMPAS_DEVICE compas::complex_type<T> F(                                   \
            unsigned long long mask,                                               \
            const compas::complex_type<T>& var,                                    \
            Ty arg,                                                                \
            int width = 32) {                                                      \
            return {::F(mask, var.re, arg, width), ::F(mask, var.im, arg, width)}; \
        }
#endif

COMPAS_COMPLEX_DEVICE_SHFL_IMPL(__shfl_sync, int);
COMPAS_COMPLEX_DEVICE_SHFL_IMPL(__shfl_up_sync, unsigned int);
COMPAS_COMPLEX_DEVICE_SHFL_IMPL(__shfl_down_sync, unsigned int);
COMPAS_COMPLEX_DEVICE_SHFL_IMPL(__shfl_xor_sync, int);