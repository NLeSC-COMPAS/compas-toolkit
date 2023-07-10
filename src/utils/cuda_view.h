#pragma once

#include <cstddef>

#include "assertion.h"
#include "macros.h"

#if COMPAS_IS_HOST
    #include <array>
#endif

namespace compas {

using index_t = int;

template<typename T, size_t N>
struct fixed_array {
    COMPAS_HOST_DEVICE
    T& operator[](index_t index) {
        COMPAS_DEBUG_ASSERT(index >= 0 && index < N);
        return items[index];
    }

    COMPAS_HOST_DEVICE
    const T& operator[](index_t index) const {
        COMPAS_DEBUG_ASSERT(index >= 0 && index < N);
        return items[index];
    }

    COMPAS_HOST_DEVICE
    T* data() {
        return &items[0];
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return &items[0];
    }

    T items[N];
};

template<typename T>
struct fixed_array<T, 0> {
    COMPAS_HOST_DEVICE
    T& operator[](index_t) {
        COMPAS_PANIC("index out of bounds");
    }

    COMPAS_HOST_DEVICE
    const T& operator[](index_t) const {
        COMPAS_PANIC("index out of bounds");
    }

    COMPAS_HOST_DEVICE
    T* data() {
        return nullptr;
    }

    COMPAS_HOST_DEVICE
    const T* data() const {
        return nullptr;
    }
};

namespace layouts {
template<size_t N>
struct row_major {
    static constexpr size_t rank = N;

    row_major() = default;

    COMPAS_HOST_DEVICE
    row_major(fixed_array<index_t, N> sizes) : sizes_(sizes) {}

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return sizes_[axis];
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        ptrdiff_t stride = 1;

        for (int i = axis + 1; i < N; i++) {
            stride *= ptrdiff_t(sizes_[i]);
        }

        return stride;
    }

  private:
    fixed_array<index_t, N> sizes_ = {0};
};

template<size_t N>
struct strided {
    static constexpr size_t rank = N;

    strided() = default;

    COMPAS_HOST_DEVICE
    strided(fixed_array<index_t, N> sizes, fixed_array<ptrdiff_t, N> strides) :
        sizes_(sizes),
        strides_(strides) {}

    COMPAS_HOST_DEVICE
    strided(const row_major<N>& that) {
        for (int i = 0; i < N; i++) {
            sizes_[i] = that.size(i);
            strides_[i] = that.stride(i);
        }
    }

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return sizes_[axis];
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        return strides_[axis];
    }

  private:
    fixed_array<index_t, N> sizes_ = {0};
    fixed_array<ptrdiff_t, N> strides_ = {0};
};

template<typename L, size_t M = 1>
struct drop_leading_axes {
    static_assert(M <= L::rank, "invalid rank of parent layout");
    static constexpr size_t rank = L::rank - M;

    COMPAS_HOST_DEVICE
    drop_leading_axes(const L& parent = {}) : parent_(parent) {}

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return parent_.size(axis + M);
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        return parent_.stride(axis + M);
    }

  private:
    L parent_;
};
}  // namespace layouts

template<typename T, typename L, size_t N = L::rank>
struct cuda_view_access;

template<typename T, typename L>
struct cuda_view_impl {
    static constexpr size_t rank = L::rank;
    using value_type = T;
    using layout_type = L;

    COMPAS_HOST_DEVICE
    cuda_view_impl(T* ptr, L layout) : ptr_(ptr), layout_(layout) {}

    COMPAS_HOST_DEVICE
    cuda_view_impl() : cuda_view_impl(nullptr, layout_type {}) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE cuda_view_impl(const cuda_view_impl<T2, L2>& that) :
        ptr_(that.ptr_),
        layout_(that.layout_) {}

    COMPAS_HOST_DEVICE
    const layout_type& layout() const {
        return layout_;
    }

    COMPAS_HOST_DEVICE
    T* data() const {
        return ptr_;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        return layout_.stride(axis);
    }

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return layout_.size(axis);
    }

    COMPAS_HOST_DEVICE
    typename cuda_view_access<T, L>::type operator[](index_t index) const {
        return cuda_view_access<T, L>::call(*this, index);
    }

  private:
    T* ptr_;
    L layout_;
};

template<typename T, typename L, size_t N>
struct cuda_view_access {
    using type = cuda_view_impl<T, layouts::drop_leading_axes<L>>;

    COMPAS_HOST_DEVICE
    static type call(const cuda_view_impl<T, L>& view, index_t index) {
        T* new_ptr = view.data() + index * view.layout().stride(0);
        layouts::drop_leading_axes<L> new_layout = view.layout();
        return {new_ptr, new_layout};
    }
};

template<typename T, typename L>
struct cuda_view_access<T, L, 1> {
    using type = T&;

    COMPAS_HOST_DEVICE
    static type call(const cuda_view_impl<T, L>& view, index_t index) {
        return view.data() + index * view.layout().stride(0);
    }
};

template<typename T, size_t N = 1>
using cuda_view = cuda_view_impl<const T, layouts::row_major<N>>;

template<typename T, size_t N = 1>
using cuda_view_mut = cuda_view_impl<T, layouts::row_major<N>>;

}  // namespace compas