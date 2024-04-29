#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "compas/core/view.h"
#include "kmm/array.hpp"
#include "kmm/cuda/cuda.hpp"
#include "kmm/host/host.hpp"
#include "kmm/runtime_handle.hpp"

#define COMPAS_CUDA_CHECK(...) KMM_CUDA_CHECK(__VA_ARGS__)

namespace compas {

template<typename T, size_t N = 1>
using Array = kmm::Array<T, N>;

struct CudaContext {
    CudaContext(kmm::RuntimeHandle runtime, kmm::Cuda device) :
        m_runtime(runtime),
        m_device(device) {}

    template<typename T, size_t N>
    Array<std::decay_t<T>, N> allocate(view_mut<T, N> content) const {
        std::array<index_t, N> sizes;
        for (size_t i = 0; i < N; i++) {
            sizes[i] = content.size(i);
        }

        return m_runtime.allocate_array(content.data(), sizes);
    }

    template<typename T, typename... Sizes>
    Array<std::decay_t<T>, sizeof...(Sizes)> allocate(const T* content_ptr, Sizes... sizes) const {
        std::array<index_t, sizeof...(Sizes)> sizes_array = {kmm::checked_cast<index_t>(sizes)...};
        return m_runtime.allocate_array(content_ptr, sizes_array);
    }

    template<typename T>
    Array<std::decay_t<T>> allocate(const std::vector<T>& content) const {
        std::array<index_t, 1> sizes_array = {kmm::checked_cast<index_t>(content.size())};
        return m_runtime.allocate_array(content.data(), sizes_array);
    }

    template<typename... Args>
    void submit_host(Args... args) const {
        m_runtime.submit(kmm::Host(), args...);
    }

    template<typename... Args>
    void submit_device(Args... args) const {
        m_runtime.submit(m_device, args...);
    }

    template<typename F, typename... Args>
    void submit_kernel(dim3 grid_dim, dim3 block_dim, F kernel, Args... args) const {
        m_runtime.submit(kmm::CudaKernel(grid_dim, block_dim, m_device), kernel, args...);
    }

    void synchronize() const {
        m_runtime.synchronize();
    }

  private:
    kmm::RuntimeHandle m_runtime;
    kmm::Cuda m_device;
};

CudaContext make_context(int device = 0) {
    return {kmm::build_runtime(), kmm::Cuda(device)};
}

}  // namespace compas