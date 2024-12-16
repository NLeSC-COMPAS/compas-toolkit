#pragma once
#include "kmm/core/view.hpp"

namespace compas {

using index_t = int;

using kmm::strided_subview;
using kmm::strided_subview_mut;
using kmm::strided_view;
using kmm::strided_view_mut;
using kmm::subview;
using kmm::subview_mut;
using kmm::view;
using kmm::view_mut;

using kmm::gpu_strided_subview;
using kmm::gpu_strided_subview_mut;
using kmm::gpu_strided_view;
using kmm::gpu_strided_view_mut;
using kmm::gpu_subview;
using kmm::gpu_subview_mut;
using kmm::gpu_view;
using kmm::gpu_view_mut;

template<typename T, size_t N = 1>
using host_view = kmm::view<T, N>;

template<typename T, size_t N = 1>
using host_view_mut = kmm::view_mut<T, N>;

}  // namespace compas