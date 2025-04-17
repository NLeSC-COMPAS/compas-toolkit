#pragma once
#include "kmm/core/view.hpp"
#include "kmm/utils/geometry.hpp"

namespace compas {

using index_t = int;

using kmm::Subview;
using kmm::SubviewMut;
using kmm::SubviewStrided;
using kmm::SubviewStridedMut;
using kmm::View;
using kmm::ViewMut;
using kmm::ViewStrided;
using kmm::ViewStridedMut;

using kmm::GPUSubview;
using kmm::GPUSubviewMut;
using kmm::GPUSubviewStrided;
using kmm::GPUSubviewStridedMut;
using kmm::GPUView;
using kmm::GPUViewMut;
using kmm::GPUViewStrided;
using kmm::GPUViewStridedMut;

template<typename T, size_t N = 1>
using HostView = kmm::View<T, N>;

template<typename T, size_t N = 1>
using HostViewMut = kmm::ViewMut<T, N>;

}  // namespace compas