#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "core/view.h"

namespace compas {

template<typename T, index_t N = 1>
struct CudaArray;
struct CudaContext;
struct CudaContextImpl;
struct CudaBuffer;
struct CudaContextGuard;

struct CudaException: public std::exception {
    CudaException(std::string msg);
    CudaException(CUresult err, const char* file, int line);
    CudaException(cudaError_t err, const char* file, int line);

    const char* what() const noexcept {
        return message_.c_str();
    }

  private:
    std::string message_;
};

#define COMPAS_CUDA_CHECK(expr)                                      \
    do {                                                             \
        auto code = (expr);                                          \
        if (code != 0) {                                             \
            throw ::compas::CudaException(code, __FILE__, __LINE__); \
        }                                                            \
    } while (0)

struct CudaContext {
    friend CudaContextGuard;

    CudaContext(std::shared_ptr<CudaContextImpl> impl) : impl_(impl) {
        COMPAS_ASSERT(impl != nullptr);
    }

    std::string device_name() const;
    std::shared_ptr<CudaBuffer> allocate_buffer(size_t nbytes) const;

    template<typename F>
    void launch(F fun) const {
        F x = 1;
        fun();
    }

    template<typename T, int N = 1>
    CudaArray<T, N> allocate(vector<index_t, N> shape) const {
        size_t nbytes = sizeof(T);
        for (index_t i = 0; i < N; i++) {
            nbytes *= size_t(shape[i]);
        }

        return {allocate_buffer(nbytes), shape};
    }

    template<typename T>
    CudaArray<T> allocate(index_t n) const {
        return allocate<T>(vector<index_t, 1> {n});
    }

    template<typename T, int N>
    CudaArray<std::decay_t<T>, N> allocate(host_view_mut<T, N> buffer) const {
        auto result = allocate<std::decay_t<T>, N>(buffer.shape());
        result.copy_from(buffer);
        return result;
    }

    template<typename T, int N = 1>
    CudaArray<T, N> zeros(vector<index_t, N> shape) const {
        auto buffer = allocate<T>(shape);
        buffer.fill(T {});
        return buffer;
    }

    template<typename T>
    CudaArray<T> zeros(index_t n) const {
        return zeros<T>({n});
    }

    template<typename T, int N>
    CudaArray<T, N> copy(const CudaArray<T, N>& input) const {
        CudaArray<T, N> result = allocate<T, N>(input.shape());
        result.copy_from(input);
        return result;
    }

  private:
    std::shared_ptr<CudaContextImpl> impl_;
};

CudaContext make_context(int device = 0);

struct CudaContextGuard {
    CudaContextGuard(std::shared_ptr<CudaContextImpl> impl);
    CudaContextGuard(const CudaContext& ctx) : CudaContextGuard(ctx.impl_) {}
    ~CudaContextGuard();

  private:
    std::shared_ptr<CudaContextImpl> impl_;
};

struct CudaBuffer {
    CudaBuffer(const CudaContext& context, size_t nbytes);
    ~CudaBuffer();

    void copy_from_host(const void* host_ptr, size_t offset, size_t nbytes);
    void copy_to_host(void* host_ptr, size_t offset, size_t nbytes);

    void copy_from_device(CUdeviceptr src_ptr, size_t offset, size_t nbytes);
    void copy_to_device(CUdeviceptr dst_ptr, size_t offset, size_t nbytes);

    void fill(const void* element_ptr, size_t element_nbytes, size_t offset, size_t nbytes);

    CUdeviceptr device_data() {
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
    CUdeviceptr device_ptr_;
    size_t nbytes_;
};

template<typename T, index_t N>
struct CudaArray {
    CudaArray(std::shared_ptr<CudaBuffer> buffer, vector<index_t, N> shape, size_t offset = 0) :
        buffer_(std::move(buffer)),
        shape_(shape),
        offset_(offset) {}

    const CudaContext& context() const {
        return buffer_->context();
    }

    index_t size(int axis) const {
        COMPAS_ASSERT(axis >= 0 && axis < N);
        return shape_[axis];
    }

    index_t size() const {
        index_t total = 1;
        for (index_t axis = 0; axis < N; axis++) {
            total *= size(axis);
        }
        return total;
    }

    size_t size_in_bytes() const {
        return size_t(size()) * sizeof(T);
    }

    vector<int, N> shape() const {
        return shape_;
    }

    template<index_t M>
    CudaArray<T, M> reshape(vector<index_t, M> new_shape) const {
        index_t total = 1;
        for (index_t i = 0; i < M; i++) {
            total *= new_shape[i];
        }

        COMPAS_ASSERT(total == size());
        return {buffer_, new_shape};
    }

    CudaArray<T> flatten() const {
        vector<index_t, 1> new_shape = {size()};
        return {buffer_, new_shape};
    }

    CudaArray<T, N> slice(index_t begin, index_t end) const {
        COMPAS_ASSERT(N > 0 && 0 <= begin && begin <= end && end <= size(0));
        size_t stride = size_t(begin);
        vector<index_t, N> new_shape;
        new_shape[0] = end - begin;

        for (int i = 0; i < N - 1; i++) {
            new_shape[i + 1] = shape_[i + 1];
            stride *= size_t(shape_[i + 1]);
        }

        return {buffer_, new_shape, offset_ + stride};
    }

    CudaArray<T, N - 1> slice(index_t index) const {
        COMPAS_ASSERT(N > 0 && index >= 0 && index < size(0));
        size_t stride = size_t(index);
        vector<index_t, N - 1> new_shape;

        for (int i = 0; i < N - 1; i++) {
            new_shape[i] = shape_[i + 1];
            stride *= size_t(shape_[i + 1]);
        }

        return {buffer_, new_shape, offset_ + stride};
    }

    const T* device_data() const {
        return (const T*)(buffer_->device_data()) + offset_;
    }

    T* device_data_mut() const {
        return (T*)(buffer_->device_data()) + offset_;
    }

    cuda_view<T, N> view() const {
        return {device_data(), shape_};
    }

    cuda_view_mut<T, N> view_mut() const {
        return {device_data_mut(), shape_};
    }

    void copy_from(host_view<T, N> input) const {
        COMPAS_ASSERT(input.shape() == shape());
        buffer_->copy_from_host(input.data(), offset_ * sizeof(T), size_in_bytes());
    }

    void copy_to(host_view_mut<T, N> output) const {
        COMPAS_ASSERT(output.shape() == shape());
        buffer_->copy_to_host(output.data(), offset_ * sizeof(T), size_in_bytes());
    }

    void copy_from(const CudaArray<T, N>& input) const {
        COMPAS_ASSERT(input.shape() == shape());
        buffer_->copy_from_device(input.data(), offset_ * sizeof(T), size_in_bytes());
    }

    void copy_to(const CudaArray<T, N>& output) const {
        return output.copy_from(*this);
    }

    void fill(const T& value) const {
        buffer_->fill(&value, sizeof(T), offset_ * sizeof(T), size_in_bytes());
    }

  private:
    std::shared_ptr<CudaBuffer> buffer_;
    vector<index_t, N> shape_;
    size_t offset_;
};

}  // namespace compas