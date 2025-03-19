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
    CompasContext(kmm::RuntimeHandle runtime, kmm::DeviceId device) :
        m_runtime(runtime),
        m_device(device) {}

    template<typename T, size_t N>
    Array<std::decay_t<T>, N> allocate(View<T, N> content) const {
        return m_runtime.allocate(content.data(), kmm::Dim<N>::from(content.sizes()));
    }

    template<typename T, typename... Sizes>
    Array<std::decay_t<T>, sizeof...(Sizes)> allocate(const T* content_ptr, Sizes... sizes) const {
        kmm::Dim<sizeof...(Sizes)> sizes_array = {kmm::checked_cast<index_t>(sizes)...};
        return m_runtime.allocate(content_ptr, sizes_array);
    }

    template<typename T>
    Array<std::decay_t<T>> allocate(const std::vector<T>& content) const {
        return m_runtime.allocate(content.data(), content.size());
    }

    template<typename L, typename... Args>
    void
    parallel_submit(kmm::DomainDim index_space, kmm::DomainDim chunk_size, L launcher, Args... args)
        const {
        m_runtime.parallel_submit(  //
            kmm::TileDomain(index_space, chunk_size),
            launcher,
            args...);
    }

    template<typename F, typename... Args>
    void
    parallel_device(kmm::DomainDim index_space, kmm::DomainDim chunk_size, F fun, Args... args) const {
        m_runtime.parallel_submit(  //
            kmm::TileDomain(index_space, chunk_size),
            kmm::GPU(fun),
            args...);
    }

    template<typename F, typename... Args>
    void parallel_kernel(
        kmm::DomainDim index_space,
        kmm::DomainDim chunk_size,
        dim3 block_dim,
        F kernel,
        Args... args) const {
        m_runtime.parallel_submit(
            kmm::TileDomain(index_space, chunk_size),
            kmm::GPUKernel(kernel, block_dim),
            args...);
    }

    template<typename F, typename... Args>
    void submit_device(kmm::DomainDim index_space, F fun, Args... args) const {
        m_runtime.submit(index_space, m_device, kmm::GPU(fun), args...);
    }

    template<typename F, typename... Args>
    void submit_kernel(dim3 grid_dim, dim3 block_dim, F kernel, Args... args) const {
        m_runtime.submit(
            kmm::DomainDim(grid_dim.x, grid_dim.y, grid_dim.z),
            m_device,
            kmm::GPUKernel(kernel, block_dim, dim3()),
            args...);
    }

    void synchronize() const {
        m_runtime.synchronize();
    }

    const kmm::RuntimeHandle& runtime() const {
        return m_runtime;
    }

  private:
    kmm::RuntimeHandle m_runtime;
    kmm::DeviceId m_device;
};

inline CompasContext make_context(int device = 0) {
    return {kmm::make_runtime(), kmm::DeviceId(device)};
}

}  // namespace compas

template<typename T>
struct kmm::DataTypeOf<compas::complex_type<T>>: kmm::DataTypeOf<::std::complex<T>> {};
