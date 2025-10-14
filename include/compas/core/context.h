#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compas/core/backends.h"
#include "compas/core/complex_type.h"
#include "compas/core/view.h"
#include "kmm/kmm.hpp"

#define COMPAS_GPU_CHECK(...) KMM_GPU_CHECK(__VA_ARGS__)

namespace compas {

template<typename T, size_t N = 1>
using Array = kmm::Array<T, N>;

struct CompasContext {
    CompasContext(kmm::Runtime& runtime, kmm::ResourceId resource_id) :
        m_runtime(kmm::RuntimeHandle(runtime).constrain_to(resource_id)),
        m_device(resource_id.as_device()) {}

    CompasContext(kmm::Runtime& runtime):
            CompasContext(runtime, kmm::DeviceId(0)) {}

    CompasContext with_device(int index) {
        auto resources = m_runtime.worker().system_info().resources();
        return {m_runtime.worker(), resources[index % resources.size()]};
    }

    template<typename T, size_t N>
    Array<std::decay_t<T>, N> allocate(View<T, N> content) const {
        return m_runtime.allocate(
            content.data(),
            kmm::Dim<N>::from(content.sizes()),
            kmm::MemoryId(m_device));
    }

    template<typename T, typename... Sizes>
    Array<std::decay_t<T>, sizeof...(Sizes)> allocate(const T* content_ptr, Sizes... sizes) const {
        kmm::Dim<sizeof...(Sizes)> sizes_array = {kmm::checked_cast<index_t>(sizes)...};
        return m_runtime.allocate(content_ptr, sizes_array, kmm::MemoryId(m_device));
    }

    template<typename T>
    Array<std::decay_t<T>> allocate(const std::vector<T>& content) const {
        return allocate(content.data(), content.size());
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
    void parallel_device(kmm::DomainDim index_space, kmm::DomainDim chunk_size, F fun, Args... args)
        const {
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
    void submit_device(F fun, Args... args) const {
        m_runtime.submit(m_device, kmm::GPU(fun), args...);
    }

    template<typename F, typename... Args>
    void submit_kernel(dim3 grid_dim, dim3 block_dim, F kernel, Args... args) const {
        auto domain = kmm::Domain {{{kmm::DomainChunk {
            .owner_id = m_device,
            .offset = {0, 0, 0},
            .size = {grid_dim.x, grid_dim.y, grid_dim.z}}}}};

        m_runtime.parallel_submit(domain, kmm::GPUKernel(kernel, block_dim, dim3()), args...);
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
    auto config = kmm::default_config_from_environment();

    // TODO: Use caching pools for host and device until segfaults are fixed
    config.host_memory_kind = kmm::HostMemoryKind::CachingPool;
    config.device_memory_kind = kmm::DeviceMemoryKind::DefaultPool;
    config.device_concurrent_streams = 4;

    // Use the caching pool instead in HIP. Memory errors occur when using the async memory pool.
#ifdef COMPAS_USE_HIP
    config.device_memory_kind = kmm::DeviceMemoryKind::CachingPool;
#endif

    return CompasContext(kmm::make_runtime(config).worker()).with_device(device);
}

}  // namespace compas

KMM_DEFINE_SCALAR_TYPE(Complex32, compas::cfloat)
KMM_DEFINE_SCALAR_TYPE(Complex64, compas::cdouble)
