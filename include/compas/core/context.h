#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "kmm/kmm.hpp"

#define COMPAS_GPU_CHECK(...) KMM_GPU_CHECK(__VA_ARGS__)

namespace compas {

template<typename T, size_t N = 1>
using Array = kmm::Array<T, N>;

struct CompasContext {
    CompasContext(kmm::Runtime runtime, kmm::DeviceId device) :
        m_runtime(runtime),
        m_device(device) {}

    template<typename T, size_t N>
    Array<std::decay_t<T>, N> allocate(host_view<T, N> content) const {
        return m_runtime.allocate(content.data(), kmm::Size<N>(content.sizes()));
    }

    template<typename T, typename... Sizes>
    Array<std::decay_t<T>, sizeof...(Sizes)> allocate(const T* content_ptr, Sizes... sizes) const {
        kmm::Size<sizeof...(Sizes)> sizes_array = {kmm::checked_cast<index_t>(sizes)...};
        return m_runtime.allocate(content_ptr, sizes_array);
    }

    template<typename T>
    Array<std::decay_t<T>> allocate(const std::vector<T>& content) const {
        return m_runtime.allocate(content.data(), content.size());
    }

    template<typename F, typename... Args>
    void submit_device(kmm::NDRange index_space, F fun, Args... args) const {
        m_runtime.submit(index_space, m_device, kmm::GPU(fun), args...);
    }

    template<typename F, typename... Args>
    void submit_kernel(dim3 grid_dim, dim3 block_dim, F kernel, Args... args) const {
        m_runtime.submit(
            kmm::NDRange(grid_dim.x, grid_dim.y, grid_dim.z),
            m_device,
            kmm::GPUKernel(kernel, block_dim, dim3()),
            args...);
    }

    void synchronize() const {
        m_runtime.synchronize();
    }

  private:
    kmm::Runtime m_runtime;
    kmm::DeviceId m_device;
};

inline CompasContext make_context(int device = 0) {
    spdlog::set_level(spdlog::level::trace);
    return {kmm::make_runtime(), kmm::DeviceId(device)};
}

}  // namespace compas

template<typename T>
struct kmm::DataTypeMap<compas::complex_type<T>>: kmm::DataTypeMap<::std::complex<T>> {};
