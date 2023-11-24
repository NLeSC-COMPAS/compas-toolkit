#include <cuda.h>

#include <iostream>
#include <stdexcept>
#include <utility>
#include <cstring>

#include "context.h"

namespace compas {

static std::string format_exception_message(CUresult err, const char* file, const int line) {
    const char* name = "";
    const char* msg = "";

    cuGetErrorName(err, &name);
    cuGetErrorString(err, &msg);

    char output[1024];
    snprintf(output, sizeof output, "CUDA error: %s (%s) at %s:%d", name, msg, file, line);

    return output;
}

static std::string format_exception_message(cudaError_t err, const char* file, const int line) {
    auto name = cudaGetErrorName(err);
    auto msg = cudaGetErrorString(err);

    char output[1024];
    snprintf(output, sizeof output, "CUDA error: %s (%s) at %s:%d", name, msg, file, line);

    return output;
}

CudaException::CudaException(CUresult err, const char* file, const int line) :
    message_(format_exception_message(err, file, line)) {
    //
}

CudaException::CudaException(cudaError_t err, const char* file, const int line) :
    message_(format_exception_message(err, file, line)) {
    //
}

CudaException::CudaException(std::string msg) : message_("CUDA error: " + msg) {
    //
}

struct CudaContextImpl {
    CudaContextImpl(CUdevice device) {
        COMPAS_CUDA_CHECK(cuInit(0));
        COMPAS_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    }

    ~CudaContextImpl() {
        try {
            COMPAS_CUDA_CHECK(cuDevicePrimaryCtxRelease(device));
        } catch (const CudaException& e) {
            std::cerr << "ignoring cuda error: " << e.what() << "\n";
        }
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
};

CudaContext make_context(int device) {
    return std::make_shared<CudaContextImpl>(device);
}

CudaContextGuard::CudaContextGuard(std::shared_ptr<CudaContextImpl> impl) : impl_(std::move(impl)) {
    COMPAS_CUDA_CHECK(cuCtxPushCurrent(impl_->context));
}

CudaContextGuard::~CudaContextGuard() noexcept(false) {
    // Synchronize first, then pop the context, then check the synchronization result. This way, the current
    // context will be popped even if `cuCtxSynchronize` returns an error.
    auto synchronize_result = cuCtxSynchronize();

    try {
        CUcontext current;
        COMPAS_CUDA_CHECK(cuCtxPopCurrent(&current));
    } catch (const CudaException& e) {
        std::cerr << "ignoring cuda error: " << e.what() << "\n";
    }

    COMPAS_CUDA_CHECK(synchronize_result);
}

std::string CudaContext::device_name() const {
    CudaContextGuard guard {*this};
    char name[512] = {0};
    COMPAS_CUDA_CHECK(cuDeviceGetName(name, sizeof(name), impl_->device));
    return name;
}

std::shared_ptr<CudaBuffer> CudaContext::allocate_buffer(size_t nbytes) const {
    return std::make_shared<CudaBuffer>(*this, nbytes);
}

void CudaContext::fill_buffer(
    CUdeviceptr output_ptr,
    size_t num_elements,
    const void* fill_value,
    size_t element_nbytes) const {
    COMPAS_ASSERT(element_nbytes > 0);
    size_t nbytes = num_elements * element_nbytes;

    bool all_equal = true;
    for (size_t i = 1; i < element_nbytes; i++) {
        if (static_cast<const char*>(fill_value)[i] != static_cast<const char*>(fill_value)[0]) {
            all_equal = false;
        }
    }

    CudaContextGuard guard {*this};

    if (all_equal || element_nbytes == 1) {
        char value = static_cast<const char*>(fill_value)[0];
        COMPAS_CUDA_CHECK(cuMemsetD8(output_ptr, value, nbytes));
    } else if (element_nbytes == 2) {
        uint16_t value;
        std::memcpy(&value, fill_value, element_nbytes);
        COMPAS_CUDA_CHECK(cuMemsetD16(output_ptr, value, nbytes));
    } else if (element_nbytes == 4) {
        uint32_t value;
        std::memcpy(&value, fill_value, element_nbytes);
        COMPAS_CUDA_CHECK(cuMemsetD32(output_ptr, value, nbytes));
    } else {
        COMPAS_PANIC("fill can only be performed using 8, 16, or 32 bit values");
    }
}

CudaBuffer::CudaBuffer(const CudaContext& context, CUdeviceptr ptr, size_t nbytes) :
    context_(context),
    is_owned_(false),
    device_ptr_(ptr),
    nbytes_(nbytes) {}

CudaBuffer::CudaBuffer(const CudaContext& context, size_t nbytes) :
    context_(context),
    is_owned_(true),
    device_ptr_(CUdeviceptr {}),
    nbytes_(0) {
    if (nbytes > 0) {
        CudaContextGuard guard {context_};
        COMPAS_CUDA_CHECK(cuMemAlloc((CUdeviceptr*)&device_ptr_, nbytes));
        nbytes_ = nbytes;
    }
}

CudaBuffer::~CudaBuffer() {
    if (nbytes_ > 0 && is_owned_) {
        try {
            CudaContextGuard guard {context_};
            COMPAS_CUDA_CHECK(cuMemFree(device_ptr_));
        } catch (const CudaException& e) {
            std::cerr << "ignoring cuda error: " << e.what() << "\n";
        }
    }
}

void CudaBuffer::copy_from_host(const void* host_ptr, size_t offset, size_t length) {
    COMPAS_ASSERT(offset <= nbytes_ && length <= nbytes_ - offset);

    CudaContextGuard guard {context_};
    COMPAS_CUDA_CHECK(cuMemcpyHtoD(device_ptr_ + offset, host_ptr, length));
}

void CudaBuffer::copy_to_host(void* host_ptr, size_t offset, size_t length) {
    COMPAS_ASSERT(offset <= nbytes_ && length <= nbytes_ - offset);

    CudaContextGuard guard {context_};
    COMPAS_CUDA_CHECK(cuMemcpyDtoH(host_ptr, device_ptr_ + offset, length));
}

void CudaBuffer::copy_from_device(CUdeviceptr src_ptr, size_t offset, size_t length) {
    COMPAS_ASSERT(offset <= nbytes_ && length <= nbytes_ - offset);

    CudaContextGuard guard {context_};
    COMPAS_CUDA_CHECK(cuMemcpyDtoD(src_ptr, device_ptr_ + offset, length));
}

void CudaBuffer::copy_to_device(CUdeviceptr dst_ptr, size_t offset, size_t length) {
    COMPAS_ASSERT(offset <= nbytes_ && length <= nbytes_ - offset);

    CudaContextGuard guard {context_};
    COMPAS_CUDA_CHECK(cuMemcpyDtoD(device_ptr_ + offset, dst_ptr, length));
}

}  // namespace compas