#include <cmath>

#include "assertion.h"

namespace compas {

using index_t = int;

template<typename T, int N>
struct vector_storage {
    COMPAS_HOST_DEVICE
    T* data() {
        return items;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return items;
    }

    T items[N];
};

template<typename T>
struct vector_storage<T, 0> {
    COMPAS_HOST_DEVICE
    T* data() {
        return nullptr;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return nullptr;
    }
};

template<typename T>
struct vector_storage<T, 1> {
    COMPAS_HOST_DEVICE
    vector_storage(T x = {}) : x(x) {}

    COMPAS_HOST_DEVICE
    T* data() {
        return items;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return items;
    }

    COMPAS_HOST_DEVICE
    operator T() const {
        return x;
    }

    union {
        T x;
        T items[1];
    };
};

template<typename T>
struct alignas(2 * alignof(T)) vector_storage<T, 2> {
    COMPAS_HOST_DEVICE
    vector_storage(T x, T y) : x(x), y(y) {}

    COMPAS_HOST_DEVICE
    vector_storage() : vector_storage(T {}, T {}) {}

    COMPAS_HOST_DEVICE
    T* data() {
        return items;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return items;
    }

    union {
        struct {
            T x;
            T y;
        };
        T items[2];
    };
};

template<typename T>
struct vector_storage<T, 3> {
    COMPAS_HOST_DEVICE
    vector_storage(T x, T y, T z) : x(x), y(y), z(z) {}

    COMPAS_HOST_DEVICE
    vector_storage() : vector_storage(T {}, T {}, T {}) {}

    COMPAS_HOST_DEVICE
    T* data() {
        return items;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return items;
    }

    union {
        struct {
            T x;
            T y;
            T z;
        };
        T items[3];
    };
};

template<typename T>
struct alignas(4 * alignof(T)) vector_storage<T, 4> {
    COMPAS_HOST_DEVICE
    vector_storage(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

    COMPAS_HOST_DEVICE
    vector_storage() : vector_storage(T {}, T {}, T {}, T {}) {}

    COMPAS_HOST_DEVICE
    T* data() {
        return items;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return items;
    }

    union {
        struct {
            T x;
            T y;
            T z;
            T w;
        };
        T items[4];
    };
};

template<typename T, int N>
struct vector: vector_storage<T, N> {
    using storage_type = vector_storage<T, N>;

    COMPAS_HOST_DEVICE
    vector(const T& value = {}) {
        for (int i = 0; i < N; i++) {
            this->at(i) = value;
        }
    }

    COMPAS_HOST_DEVICE
    vector(const storage_type& storage) : storage_type(storage) {}

    template<typename... Ts>
    COMPAS_HOST_DEVICE vector(const T& first, const T& second, const Ts&... rest) :
        storage_type(first, second, rest...) {}

    COMPAS_HOST_DEVICE
    T& at(index_t i) {
        COMPAS_DEBUG_ASSERT(i >= 0 && i < N);
        return this->data()[i];
    }

    COMPAS_HOST_DEVICE
    const T& at(index_t i) const {
        COMPAS_DEBUG_ASSERT(i >= 0 && i < N);
        return this->data()[i];
    }

    COMPAS_HOST_DEVICE
    T& operator[](index_t i) {
        return at(i);
    }

    COMPAS_HOST_DEVICE
    const T& operator[](index_t i) const {
        return at(i);
    }

    COMPAS_HOST_DEVICE
    index_t size() const {
        return N;
    }

    COMPAS_HOST_DEVICE
    T* begin() {
        return this->data();
    }

    COMPAS_HOST_DEVICE
    T* end() {
        return this->data() + N;
    }

    COMPAS_HOST_DEVICE
    const T* begin() const {
        return this->data();
    }

    COMPAS_HOST_DEVICE
    const T* end() const {
        return this->data() + N;
    }

    COMPAS_HOST_DEVICE
    const T* cbegin() const {
        return this->data();
    }

    COMPAS_HOST_DEVICE
    const T* cend() const {
        return this->data() + N;
    }

    template<typename F>
    COMPAS_HOST_DEVICE auto map(F fun) -> vector<decltype(fun(T {})), N> const {
        using R = decltype(fun(T {}));
        vector<R, N> result;

        for (index_t i = 0; i < N; i++) {
            result[i] = fun(this->at(i));
        }

        return result;
    }
};

namespace detail {
template<typename T, index_t N>
struct vector_norm_impl;

template<index_t N>
struct vector_norm_impl<float, N> {
    COMPAS_HOST_DEVICE
    static float call(const vector<float, N>& input) {
        float result = {};
        for (int i = 0; i < N; i++) {
            result += input[i] * input[i];
        }
        return sqrtf(result);
    }
};

template<index_t N>
struct vector_norm_impl<double, N> {
    COMPAS_HOST_DEVICE
    static double call(const vector<double, N>& input) {
        double result = {};
        for (int i = 0; i < N; i++) {
            result += input[i] * input[i];
        }
        return sqrt(result);
    }
};

template<typename T>
struct vector_norm_impl<T, 1> {
    COMPAS_HOST_DEVICE
    static T call(const vector<T, 1>& input) {
        T result = input[0];
        return fabs(result);
    }
};

template<>
struct vector_norm_impl<float, 2> {
    COMPAS_HOST_DEVICE
    static float call(const vector<float, 2>& input) {
        return hypotf(input[0], input[1]);
    }
};

template<>
struct vector_norm_impl<double, 2> {
    COMPAS_HOST_DEVICE
    static double call(const vector<double, 2>& input) {
        return hypot(input[0], input[1]);
    }
};
}  // namespace detail

template<typename T, int N>
COMPAS_HOST_DEVICE T norm(const vector<T, N>& input) {
    return detail::vector_norm_impl<T, N>::call(input);
}

template<typename T, int N>
COMPAS_HOST_DEVICE T dot(const vector<T, N>& left, const vector<T, N>& right) {
    T result = {};
    for (int i = 0; i < N; i++) {
        result += left[i] * right[i];
    }
    return result;
}

template<typename T>
COMPAS_HOST_DEVICE vector<T, 3> cross(const vector<T, 3>& left, const vector<T, 3>& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]};
}

#define COMPAS_VECTOR_OPS(OP, OP_ASSIGN)                                                        \
    template<typename T, int N>                                                                 \
    COMPAS_HOST_DEVICE vector<T, N> operator OP(const vector<T, N>& left, vector<T, N> right) { \
        vector<T, N> result;                                                                    \
        for (int i = 0; i < N; i++) {                                                           \
            result[i] = left[i] OP right[i];                                                    \
        }                                                                                       \
                                                                                                \
        return result;                                                                          \
    }                                                                                           \
                                                                                                \
    template<typename T, int N>                                                                 \
    COMPAS_HOST_DEVICE vector<T, N> operator OP(const vector<T, N>& left, const T& right) {     \
        return left OP vector<T, N> {right};                                                    \
    }                                                                                           \
                                                                                                \
    template<typename T, int N>                                                                 \
    COMPAS_HOST_DEVICE vector<T, N> operator OP(const T& left, const vector<T, N>& right) {     \
        return vector<T, N> {left} OP right;                                                    \
    }                                                                                           \
                                                                                                \
    template<typename T, int N>                                                                 \
    COMPAS_HOST_DEVICE vector<T, N> operator OP_ASSIGN(                                         \
        vector<T, N>& left,                                                                     \
        const vector<T, N>& right) {                                                            \
        return left = left OP right;                                                            \
    }                                                                                           \
                                                                                                \
    template<typename T, int N>                                                                 \
    COMPAS_HOST_DEVICE vector<T, N> operator OP_ASSIGN(vector<T, N>& left, const T& right) {    \
        return left = left OP right;                                                            \
    }

COMPAS_VECTOR_OPS(+, +=)
COMPAS_VECTOR_OPS(-, -=)
COMPAS_VECTOR_OPS(*, *=)
COMPAS_VECTOR_OPS(/, /=)

template<typename T, index_t N, typename R, index_t M>
COMPAS_HOST_DEVICE bool operator==(const vector<T, N>& left, const vector<R, M>& right) {
    if (N != M) {
        return false;
    }

    for (index_t i = 0; i < N; i++) {
        if (left[i] != right[i]) {
            return false;
        }
    }

    return true;
}

template<typename T, index_t N, typename R, index_t M>
COMPAS_HOST_DEVICE bool operator!=(const vector<T, N>& left, const vector<R, M>& right) {
    return !operator==(left, right);
}

template<typename T, int N>
using vec = vector<T, N>;

template<typename T>
using vec2 = vector<T, 2>;
using vfloat2 = vec2<float>;
using vdouble2 = vec2<double>;

template<typename T>
using vec3 = vector<T, 3>;
using vfloat3 = vec3<float>;
using vdouble3 = vec3<double>;

template<typename T>
using vec4 = vector<T, 4>;
using vfloat4 = vec4<float>;
using vdouble4 = vec4<double>;

template<int N>
using ndindex_t = vector<int, N>;

}  // namespace compas