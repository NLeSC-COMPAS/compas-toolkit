#pragma once

#include <cstddef>

#include "assertion.h"
#include "macros.h"

#if COMPAS_IS_HOST
    #include <array>
#endif

namespace compas {

using index_t = int;

template<typename T, index_t N>
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

template<typename T, index_t N, typename R, index_t M>
COMPAS_HOST_DEVICE bool
operator==(const fixed_array<T, N>& left, const fixed_array<R, M>& right) {
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
COMPAS_HOST_DEVICE bool
operator!=(const fixed_array<T, N>& left, const fixed_array<R, M>& right) {
    return !operator==(left, right);
}

namespace layouts {
template<index_t N>
struct col_major;

template<index_t N>
struct row_major {
    static constexpr index_t rank = N;
    using transpose_type = col_major<N>;

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

    COMPAS_HOST_DEVICE
    transpose_type transpose() const {
        return sizes_;
    }

  private:
    fixed_array<index_t, N> sizes_ = {0};
};

template<index_t N>
struct col_major {
    static constexpr index_t rank = N;
    using transpose_type = row_major<N>;

    col_major() = default;

    COMPAS_HOST_DEVICE
    col_major(fixed_array<index_t, N> sizes) : sizes_(sizes) {}

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return sizes_[axis];
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        ptrdiff_t stride = 1;

        for (int i = 0; i < axis && i < N; i++) {
            stride *= ptrdiff_t(sizes_[i]);
        }

        return stride;
    }

    COMPAS_HOST_DEVICE
    transpose_type transpose() const {
        return sizes_;
    }

  private:
    fixed_array<index_t, N> sizes_ = {0};
};

template<index_t N>
struct strided {
    static constexpr index_t rank = N;
    using transpose_type = strided<N>;

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
    strided(const col_major<N>& that) {
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

    COMPAS_HOST_DEVICE
    transpose_type transpose() const {
        fixed_array<index_t, N> rev_sizes;
        fixed_array<ptrdiff_t, N> rev_strides;

        for (int i = 0; i < N; i++) {
            rev_sizes[i] = sizes_[N - i - 1];
            rev_strides[i] = strides_[N - i - 1];
        }

        return {rev_sizes, rev_strides};
    }

  private:
    fixed_array<index_t, N> sizes_ = {0};
    fixed_array<ptrdiff_t, N> strides_ = {0};
};

template<typename L, index_t A = 0>
struct drop_axis {
    static_assert(A < L::rank, "invalid rank of parent layout");
    static constexpr index_t rank = L::rank - 1;
    using transpose_type = drop_axis<typename L::transpose_type, rank - A - 1>;

    COMPAS_HOST_DEVICE
    drop_axis(const L& parent = {}) : parent_(parent) {}

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return parent_.size(axis + index_t(axis >= A));
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        return parent_.stride(axis + index_t(axis >= A));
    }

    COMPAS_HOST_DEVICE
    transpose_type transpose() const {
        return parent_.transpose();
    }

  private:
    L parent_;
};
}  // namespace layouts

enum struct memory_space { HOST, CUDA };

template<
    typename T,
    typename L,
    memory_space M = memory_space::HOST,
    index_t N = L::rank>
struct view_impl;

template<typename T, typename L, memory_space M>
struct view_base {
    static constexpr index_t rank = L::rank;
    static constexpr memory_space space = M;
    using value_type = T;
    using layout_type = L;

    COMPAS_HOST_DEVICE
    view_base(T* ptr, L layout) : ptr_(ptr), layout_(layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE view_base(const view_base<T2, L2, M>& that) :
        ptr_(that.data()),
        layout_(that.layout()) {}

    COMPAS_HOST_DEVICE
    T* data() const {
        return ptr_;
    }

    COMPAS_HOST_DEVICE
    const layout_type& layout() const {
        return layout_;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t stride(index_t axis) const {
        return layout_.stride(axis);
    }

    fixed_array<ptrdiff_t, rank> strides() const {
        fixed_array<ptrdiff_t, rank> result;
        for (index_t i = 0; i < rank; i++) {
            result[i] = stride(i);
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return layout_.size(axis);
    }

    fixed_array<index_t, rank> shape() const {
        fixed_array<index_t, rank> result;
        for (index_t i = 0; i < rank; i++) {
            result[i] = size(i);
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    index_t size() const {
        index_t total = 1;
        for (index_t axis = 0; axis < rank; axis++) {
            total *= layout_.size(axis);
        }

        return total;
    }

    COMPAS_HOST_DEVICE
    bool is_empty() const {
        return size() == 0;
    }

    COMPAS_HOST_DEVICE
    view_impl<T, layouts::drop_axis<L>, M>
    drop_leading_axis(index_t index = 0) const {
        COMPAS_DEBUG_ASSERT(index >= 0 && index < size(0));
        ptrdiff_t offset = stride(0) * index;
        return {ptr_ + offset, layout_};
    };

    COMPAS_HOST_DEVICE
    view_impl<T, typename L::transpose_type, M> transpose() const {
        return {ptr_, layout_.transpose()};
    }

  private:
    T* ptr_;
    L layout_;
};

template<typename T, typename L, memory_space M, index_t N>
struct view_impl: view_base<T, L, M> {
    using base_type = view_base<T, L, M>;

    COMPAS_HOST_DEVICE
    view_impl(T* ptr = nullptr, L layout = {}) : base_type(ptr, layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE view_impl(const view_base<T2, L2, M>& that) :
        base_type(that) {}

    COMPAS_HOST_DEVICE
    view_impl<T, layouts::drop_axis<L>, M> operator[](index_t index) const {
        return this->drop_leading_axis(index);
    }
};

template<typename T, typename L>
struct view_impl<T, L, memory_space::HOST, 1>:
    view_base<T, L, memory_space::HOST> {
    using base_type = view_base<T, L, memory_space::HOST>;

    COMPAS_HOST_DEVICE
    view_impl(T* ptr = nullptr, L layout = {}) : base_type(ptr, layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE
    view_impl(const view_base<T2, L2, memory_space::HOST>& that) :
        base_type(that) {}

    T& operator[](index_t index) const {
        return *(this->drop_leading_axis(index).data());
    }
};

template<typename T, typename L>
struct view_impl<T, L, memory_space::CUDA, 1>:
    view_base<T, L, memory_space::CUDA> {
    using base_type = view_base<T, L, memory_space::CUDA>;

    COMPAS_HOST_DEVICE
    view_impl(T* ptr = nullptr, L layout = {}) : base_type(ptr, layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE
    view_impl(const view_base<T2, L2, memory_space::CUDA>& that) :
        base_type(that) {}

    COMPAS_DEVICE
    T& operator[](index_t index) const {
        return *(this->drop_leading_axis(index).data());
    }
};

template<typename T, index_t N = 1>
using cuda_view = view_impl<const T, layouts::row_major<N>, memory_space::CUDA>;

template<typename T, index_t N = 1>
using cuda_view_mut = view_impl<T, layouts::row_major<N>, memory_space::CUDA>;

template<typename T, index_t N = 1>
using cuda_strided_view =
    view_impl<const T, layouts::strided<N>, memory_space::CUDA>;

template<typename T, index_t N = 1>
using cuda_strided_view_mut =
    view_impl<T, layouts::strided<N>, memory_space::CUDA>;

template<typename T, index_t N = 1>
using view = view_impl<const T, layouts::row_major<N>, memory_space::HOST>;

template<typename T, index_t N = 1>
using view_mut = view_impl<T, layouts::row_major<N>, memory_space::HOST>;

template<typename T, index_t N = 1>
using host_view = view<T, N>;

template<typename T, index_t N = 1>
using host_view_mut = view_mut<T, N>;

template<typename T, index_t N = 1>
using strided_view =
    view_impl<const T, layouts::strided<N>, memory_space::HOST>;

template<typename T, index_t N = 1>
using strided_view_mut = view_impl<T, layouts::strided<N>, memory_space::HOST>;

template<typename T, index_t N = 1>
using fortran_view =
    view_impl<const T, layouts::col_major<N>, memory_space::HOST>;

template<typename T, index_t N = 1>
using fortran_view_mut =
    view_impl<T, layouts::col_major<N>, memory_space::HOST>;

}  // namespace compas