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

namespace layouts {
template<int N>
struct RowMajor {
    COMPAS_HOST_DEVICE
    RowMajor(StaticArray<int, N> shape) : shape_(shape) {}

    COMPAS_HOST_DEVICE
    RowMajor() {
        for (int i = 0; i < N; i++) {
            shape_[i] = 0;
        }
    }

    COMPAS_HOST_DEVICE
    int size(int axis) const {
        return axis < N ? shape_[axis] : 1;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(int axis) const {
        ptrdiff_t stride = 1;

        for (int i = axis + 1; i < N; i++) {
            stride *= shape_[i];
        }

        return stride;
    }

  private:
    StaticArray<int, N> shape_;
};

template<int N>
struct ColumnMajor {
    COMPAS_HOST_DEVICE
    ColumnMajor(StaticArray<int, N> shape) : shape_(shape) {}

    COMPAS_HOST_DEVICE
    int size(int axis) const {
        return axis < N ? shape_[axis] : 1;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(int axis) const {
        ptrdiff_t stride = 1;

        for (int i = 0; i < axis && i < N; i++) {
            stride *= shape_[i];
        }

        return stride;
    }

  private:
    StaticArray<int, N> shape_;
};

template<int N>
struct Strided {
    COMPAS_HOST_DEVICE
    Strided(StaticArray<int, N> shape, StaticArray<ptrdiff_t, N> strides) :
        shape_(shape),
        strides_(strides) {}

    COMPAS_HOST_DEVICE
    int size(int axis) const {
        return axis < N ? shape_[axis] : 1;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(int axis) const {
        return axis < N ? strides_[axis] : 1;
    }

  private:
    StaticArray<int, N> shape_;
    StaticArray<ptrdiff_t, N> strides_;
};
}  // namespace layouts

template<typename T, int N = 1, template<int> class L = layouts::RowMajor>
struct CudaView;

namespace detail {
template<typename T, size_t N>
struct CudaViewIndexHelper {
    using type = CudaView<T, N - 1, layouts::Strided>;

    template<template<int> class L>
    COMPAS_HOST_DEVICE static type
    call(const CudaView<T, N, L>& view, ptrdiff_t index) {
        T* base_ptr = view.data() + view.stride(0) * index;
        StaticArray<int, N - 1> new_shape;
        StaticArray<ptrdiff_t, N - 1> new_strides;

        for (int i = 1; i < N; i++) {
            new_shape[i - 1] = view.size(i);
            new_strides[i - 1] = view.stride(i);
        }

        return {base_ptr, {new_shape, new_strides}};
    }
};

template<typename T>
struct CudaViewIndexHelper<T, 1> {
    using type = T&;

    template<template<int> class L>
    COMPAS_HOST_DEVICE static type
    call(CudaView<T, 1, L> view, ptrdiff_t index) {
        return (view.data())[view.stride(0) * index];
    }
};
}  // namespace detail

template<typename T, int N, template<int> class L>
struct CudaView {
    using value_type = T;
    using layout_type = L<N>;
    static constexpr int rank = N;

    COMPAS_HOST_DEVICE
    CudaView(T* ptr, layout_type layout) : ptr_(ptr), layout_(layout) {}

    COMPAS_HOST_DEVICE
    CudaView() : CudaView(nullptr, layout_type {}) {}

    template<typename U = const T>
    operator CudaView<U, N, L>() const {
        return {ptr_, layout_};
    }

    const layout_type& layout() const {
        return layout_;
    }

    COMPAS_HOST_DEVICE
    int size(int axis) const {
        return layout_.size(axis);
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(int axis) const {
        return layout_.stride(axis);
    }

    COMPAS_HOST_DEVICE
    T* data() const {
        return ptr_;
    }

    COMPAS_HOST_DEVICE
    bool is_empty() {
        bool result = false;

        for (int i = 0; i < N; i++) {
            result |= size(i) == 0;
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    int size() const {
        int total = 0;

        for (int i = 0; i < N; i++) {
            total *= size(i);
        }

        return total;
    }

    COMPAS_HOST_DEVICE
    StaticArray<int, N> shape() const {
        StaticArray<int, N> result;

        for (int i = 0; i < N; i++) {
            result[i] = size(i);
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    typename detail::CudaViewIndexHelper<T, N>::type
    operator[](ptrdiff_t index) const {
        return detail::CudaViewIndexHelper<T, N>::call(*this, index);
    }

    T* ptr_;
    layout_type layout_;
};

}  // namespace compas