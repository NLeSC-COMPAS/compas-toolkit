#pragma once
#include <cstddef>

#include "defines.h"

namespace compas {
template<typename T, size_t N>
struct StaticArray {
    COMPAS_HOST_DEVICE
    T* data() {
        return &items[0];
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return &items[0];
    }

    COMPAS_HOST_DEVICE
    size_t size() const {
        return N;
    }

    COMPAS_HOST_DEVICE
    T& operator[](size_t i) {
        return items[i];
    }

    COMPAS_HOST_DEVICE
    const T& operator[](size_t i) const {
        return items[i];
    }

    T items[N];
};

template<typename T>
struct StaticArray<T, 0> {
    COMPAS_HOST_DEVICE
    T* data() {
        return nullptr;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return nullptr;
    }

    COMPAS_HOST_DEVICE
    size_t size() const {
        return 0;
    }

    COMPAS_HOST_DEVICE
    T& operator[](size_t i) {
        COMPAS_UNREACHABLE;
    }

    COMPAS_HOST_DEVICE
    const T& operator[](size_t i) const {
        COMPAS_UNREACHABLE;
    }
};

template<typename T, int N = 1>
struct CudaView;

namespace detail {
template<typename T, size_t N>
struct CudaViewIndexHelper {
    using type = CudaView<T, N - 1>;

    COMPAS_HOST_DEVICE
    static type call(CudaView<T, N> view, ptrdiff_t index) {
        StaticArray<int, N - 1> new_shape;
        for (int i = 1; i < N; i++) {
            new_shape[i - 1] = view.shape[i];
        }

        return {view.ptr + view.stride(0) * index, new_shape};
    }
};

template<typename T>
struct CudaViewIndexHelper<T, 1> {
    using type = T&;

    COMPAS_HOST_DEVICE
    static type call(CudaView<T, 1> view, ptrdiff_t index) {
        return view.ptr[view.stride(0) * index];
    }
};
}  // namespace detail

template<typename T, int N>
struct CudaView {
    operator CudaView<const T, N>() const {
        return {ptr, shape};
    }

    COMPAS_HOST_DEVICE
    bool is_empty() {
        bool result = false;

        for (int i = 0; i < N; i++) {
            result |= shape[i] == 0;
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    int size() const {
        int total = 0;

        for (int i = 0; i < N; i++) {
            total *= shape[i];
        }

        return total;
    }

    COMPAS_HOST_DEVICE
    int size(int axis) const {
        return axis < N ? shape[axis] : 1;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(int axis) const {
        ptrdiff_t stride = 1;

        for (int i = axis + 1; i < N; i++) {
            stride *= shape[i];
        }

        return stride;
    }

    COMPAS_HOST_DEVICE
    T* data() const {
        return ptr;
    }

    COMPAS_HOST_DEVICE
    typename detail::CudaViewIndexHelper<T, N>::type
    operator[](ptrdiff_t index) const {
        return detail::CudaViewIndexHelper<T, N>::call(*this, index);
    }

    T* ptr;
    StaticArray<int, N> shape;
};

template<typename T, typename U>
__device__ T& operator+=(CudaView<T, 1>& v, U&& value) {
    (*v.ptr) += value;
}

template<typename T, typename U>
__device__ T& operator-=(CudaView<T, 1>& v, U&& value) {
    (*v.ptr) -= value;
}

template<typename T, typename U>
__device__ T& operator/=(CudaView<T, 1>& v, U&& value) {
    (*v.ptr) /= value;
}

template<typename T, typename U>
__device__ T& operator*=(CudaView<T, 1>& v, U&& value) {
    (*v.ptr) *= value;
}

}  // namespace compas