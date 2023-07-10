#pragma once

#include <memory>

#include "utils/cuda_view.h"

namespace compas {
struct CudaContextImpl;

struct CudaContext {
  private:
    std::shared_ptr<CudaContextImpl> impl_;
};

CudaContext make_context(int device = 0);

struct CudaBuffer {
    void* device_data() {
        return device_ptr_;
    }

    const CudaContext& context() const {
        return context_;
    }

    size_t size_in_bytes() const {
        return nbytes_;
    }

  private:
    CudaContext context_;
    void* device_ptr_;
    size_t nbytes_;
};

template<typename T, size_t N = 1>
struct CudaArray {
    const CudaContext& context() const {
        return buffer_->context();
    }

    index_t size() const {
        index_t total = 1;
        for (int axis = 0; axis < N; axis++) {
            total *= shape_[axis];
        }
        return total;
    }

    index_t size(int axis) const {
        COMPAS_ASSERT(axis >= 0 && axis < N);
        return shape_[axis];
    }

    std::array<int, N> sizes() const {
        return shape_;
    }

    cuda_view<T, N> view() const {
        return {static_cast<const T*>(buffer_->device_data()), shape_};
    }

    cuda_view_mut<T, N> view_mut() const {
        return {static_cast<T*>(buffer_->device_data()), shape_};
    }

  private:
    std::shared_ptr<CudaBuffer> buffer_;
    std::array<int, N> shape_;
};

}  // namespace compas