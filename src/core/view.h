#pragma once

#include "assertion.h"
#include "macros.h"
#include "vector.h"

#if COMPAS_IS_HOST
    #include <array>
    #include <cstddef>
#endif

namespace compas {

namespace layouts {
template<index_t N>
struct col_major;

template<index_t N>
struct row_major {
    static constexpr index_t rank = N;
    using transpose_type = col_major<N>;

    row_major() = default;

    COMPAS_HOST_DEVICE
    row_major(ndindex_t<N> sizes) : sizes_(sizes) {}

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
    ndindex_t<N> sizes_ = {0};
};

template<index_t N>
struct col_major {
    static constexpr index_t rank = N;
    using transpose_type = row_major<N>;

    col_major() = default;

    COMPAS_HOST_DEVICE
    col_major(ndindex_t<N> sizes) : sizes_(sizes) {}

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
    ndindex_t<N> sizes_ = {0};
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

template<index_t N>
struct strided {
    static constexpr index_t rank = N;
    using transpose_type = strided<N>;

    strided() = default;

    COMPAS_HOST_DEVICE
    strided(ndindex_t<N> sizes, vector<ptrdiff_t, N> strides) : sizes_(sizes), strides_(strides) {}

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

    template<typename L, index_t A>
    COMPAS_HOST_DEVICE strided(const drop_axis<L, A>& that) {
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
        ndindex_t<N> rev_sizes;
        vector<ptrdiff_t, N> rev_strides;

        for (int i = 0; i < N; i++) {
            rev_sizes[i] = sizes_[N - i - 1];
            rev_strides[i] = strides_[N - i - 1];
        }

        return {rev_sizes, rev_strides};
    }

  private:
    ndindex_t<N> sizes_ = {0};
    vector<ptrdiff_t, N> strides_ = {0};
};
}  // namespace layouts

template<typename L, int Axis = 0>
struct drop_axis_impl {
    using type = layouts::drop_axis<L, Axis>;

    COMPAS_HOST_DEVICE
    static type call(const L& layout) {
        return layout;
    }
};

template<int N>
struct drop_axis_impl<layouts::row_major<N>, 0> {
    using type = layouts::row_major<N - 1>;

    COMPAS_HOST_DEVICE
    static type call(const layouts::row_major<N>& layout) {
        ndindex_t<N - 1> sizes;
        for (int i = 0; i + 1 < N; i++) {
            sizes[i] = layout.size(i + 1);
        }
        return sizes;
    }
};

enum struct memory_space { HOST, CUDA };

template<typename T, typename L, memory_space M = memory_space::HOST, index_t N = L::rank>
struct basic_view;

template<typename T, typename L, memory_space M>
struct basic_view_base {
    static constexpr index_t rank = L::rank;
    static constexpr memory_space space = M;
    using value_type = T;
    using layout_type = L;
    using ndindex_type = ndindex_t<rank>;
    using strides_type = vector<ptrdiff_t, rank>;
    using shape_type = vector<index_t, rank>;

    COMPAS_HOST_DEVICE
    basic_view_base(T* ptr, L layout) : ptr_(ptr), layout_(layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE basic_view_base(const basic_view_base<T2, L2, M>& that) :
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

    strides_type strides() const {
        strides_type result;

        for (index_t i = 0; i < rank; i++) {
            result[i] = stride(i);
        }

        return result;
    }

    COMPAS_HOST_DEVICE
    index_t size(index_t axis) const {
        return layout_.size(axis);
    }

    shape_type shape() const {
        ndindex_t<rank> result;

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

    template<index_t Axis>
    COMPAS_HOST_DEVICE basic_view<T, typename drop_axis_impl<L, Axis>::type, M>
    drop_axis(index_t index = 0) const {
        static_assert(Axis < rank, "axis out of bounds");
        COMPAS_DEBUG_ASSERT(index >= 0 && index < size(Axis));
        ptrdiff_t offset = stride(Axis) * index;
        return {ptr_ + offset, drop_axis_impl<L, Axis>::call(layout_)};
    };

    COMPAS_HOST_DEVICE
    basic_view<T, typename drop_axis_impl<L>::type, M> drop_leading_axis(index_t index = 0) const {
        return this->template drop_axis<0>(index);
    };

    COMPAS_HOST_DEVICE
    basic_view<T, typename L::transpose_type, M> transpose() const {
        return {ptr_, layout_.transpose()};
    }

    COMPAS_HOST_DEVICE
    bool in_bounds(ndindex_type indices) const {
        bool valid = true;

        for (index_t i = 0; i < rank; i++) {
            valid = valid && (indices[i] >= 0 && indices[i] < size(i));
        }

        return valid;
    }

    COMPAS_HOST_DEVICE
    ptrdiff_t linearize_index(ndindex_type indices) const {
        COMPAS_DEBUG_ASSERT(in_bounds(indices));
        ptrdiff_t offset = 0;

        for (index_t i = 0; i < rank; i++) {
            offset += ptrdiff_t(indices[i]) * layout_.stride(i);
        }

        return offset;
    }

    COMPAS_HOST_DEVICE
    T& at(ndindex_type indices) const {
        T* ptr = ptr_ + linearize_index(indices);

        if (
#if COMPAS_IS_HOST
            M == memory_space::HOST
#elif COMPAS_IS_DEVICE
            M == memory_space::CUDA
#else
            0
#endif
        ) {
            return *ptr;
        } else {
            COMPAS_PANIC("cannot dereference data from invalid memory space");
        }
    }

  private:
    T* ptr_;
    L layout_;
};

template<typename T, typename L, memory_space M, index_t N>
struct basic_view: basic_view_base<T, L, M> {
    using base_type = basic_view_base<T, L, M>;

    COMPAS_HOST_DEVICE
    basic_view(T* ptr = nullptr, L layout = {}) : base_type(ptr, layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE basic_view(const basic_view_base<T2, L2, M>& that) : base_type(that) {}

    COMPAS_HOST_DEVICE
    basic_view<T, typename drop_axis_impl<L>::type, M> operator[](index_t index) const {
        return this->drop_leading_axis(index);
    }

    COMPAS_HOST_DEVICE
    T& operator[](ndindex_t<N> indices) const {
        return this->at(indices);
    }
};

template<typename T, typename L, memory_space M>
struct basic_view<T, L, M, 1>: basic_view_base<T, L, M> {
    using base_type = basic_view_base<T, L, M>;

    COMPAS_HOST_DEVICE
    basic_view(T* ptr = nullptr, L layout = {}) : base_type(ptr, layout) {}

    template<typename T2, typename L2>
    COMPAS_HOST_DEVICE basic_view(const basic_view_base<T2, L2, M>& that) : base_type(that) {}

    COMPAS_HOST_DEVICE
    T& operator[](index_t index) const {
        return this->at(ndindex_t<1> {index});
    }

    COMPAS_HOST_DEVICE
    T& operator[](ndindex_t<1> indices) const {
        return this->at(indices);
    }
};

template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using view = basic_view<const T, layouts::row_major<N>, M>;
template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using view_mut = basic_view<T, layouts::row_major<N>, M>;

template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using strided_view = basic_view<const T, layouts::strided<N>, M>;
template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using strided_view_mut = basic_view<T, layouts::strided<N>, M>;

template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using fortran_view = basic_view<const T, layouts::col_major<N>, M>;
template<typename T, index_t N = 1, memory_space M = memory_space::HOST>
using fortran_view_mut = basic_view<T, layouts::col_major<N>, M>;

template<typename T, index_t N = 1>
using host_view = view<T, N, memory_space::HOST>;
template<typename T, index_t N = 1>
using host_view_mut = view_mut<T, N, memory_space::HOST>;

template<typename T, index_t N = 1>
using cuda_view = view<T, N, memory_space::CUDA>;
template<typename T, index_t N = 1>
using cuda_view_mut = view_mut<T, N, memory_space::CUDA>;

template<typename T, index_t N = 1>
using cuda_strided_view = strided_view<T, N, memory_space::CUDA>;
template<typename T, index_t N = 1>
using cuda_strided_view_mut = strided_view_mut<T, N, memory_space::CUDA>;

}  // namespace compas