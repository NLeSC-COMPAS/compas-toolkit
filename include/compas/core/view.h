#pragma once
#include "kmm/core/view.hpp"

namespace compas {

using index_t = int;

template<typename T, size_t N = 1>
using view = kmm::view<T, N>;

template<typename T, size_t N = 1>
using view_mut = kmm::view_mut<T, N>;

template<typename T, size_t N = 1>
using host_view = kmm::view<T, N>;

template<typename T, size_t N = 1>
using host_view_mut = kmm::view_mut<T, N>;

template<typename T, size_t N = 1>
using gpu_view = kmm::gpu_view<T, N>;

template<typename T, size_t N = 1>
using gpu_view_mut = kmm::gpu_view_mut<T, N>;

template<typename T, size_t N = 1>
using gpu_subview_mut = kmm::gpu_subview_mut<T, N>;

template<typename T, size_t N = 1>
using gpu_strided_view = kmm::gpu_strided_view<T, N>;

template<typename T, size_t N = 1>
using gpu_strided_view_mut = kmm::gpu_strided_view_mut<T, N>;

}  // namespace compas