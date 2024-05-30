#pragma once
#include "kmm/kmm.hpp"

namespace compas {

using index_t = kmm::default_index_type;

template<typename T, size_t N = 1>
using view = kmm::view<T, N>;

template<typename T, size_t N = 1>
using view_mut = kmm::view_mut<T, N>;

template<typename T, size_t N = 1>
using host_view = kmm::view<T, N>;

template<typename T, size_t N = 1>
using host_view_mut = kmm::view_mut<T, N>;

template<typename T, size_t N = 1>
using cuda_view = kmm::cuda_view<T, N>;

template<typename T, size_t N = 1>
using cuda_view_mut = kmm::cuda_view_mut<T, N>;

template<typename T, size_t N = 1>
using cuda_strided_view = kmm::cuda_strided_view<T, N>;

template<typename T, size_t N = 1>
using cuda_strided_view_mut = kmm::cuda_strided_view_mut<T, N>;

}  // namespace compas