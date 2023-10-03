
#include <cuda.h>

#include <iostream>
#include <stdexcept>
#include <utility>

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

static std::string format_exception_message(cublasStatus_t err, const char* file, const int line) {
    const char* msg = [=]() {
        switch (err) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "unknown error";
        }
    }();

    char output[1024];
    snprintf(
        output,
        sizeof output,
        "cuBLAS error: %s (code: %d) at %s:%d",
        msg,
        int(err),
        file,
        line);
    return output;
}

CudaException::CudaException(const CUresult& err, const char* file, const int line) :
    message_(format_exception_message(err, file, line)) {
    //
}

CudaException::CudaException(const cudaError_t& err, const char* file, const int line) :
    message_(format_exception_message(err, file, line)) {
    //
}

CudaException::CudaException(const cublasStatus_t& err, const char* file, const int line) :
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

        COMPAS_CUDA_CHECK(cuCtxPushCurrent(context));
        COMPAS_CUDA_CHECK(cublasCreate_v2(&cublas_handle));
        COMPAS_CUDA_CHECK(cuCtxPopCurrent(&context));
    }

    ~CudaContextImpl() {
        try {
            COMPAS_CUDA_CHECK(cublasDestroy_v2(cublas_handle));
            cublas_handle = nullptr;
        } catch (const CudaException& e) {
            std::cerr << "ignoring error during shutdown: " << e.what() << "\n";
        }

        try {
            COMPAS_CUDA_CHECK(cuDevicePrimaryCtxRelease(device));
        } catch (const CudaException& e) {
            std::cerr << "ignoring error during shutdown: " << e.what() << "\n";
        }
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    cublasHandle_t cublas_handle = nullptr;
};

CudaContext make_context(int device) {
    return std::make_shared<CudaContextImpl>(device);
}

CudaContextGuard::CudaContextGuard(std::shared_ptr<CudaContextImpl> impl) : impl_(std::move(impl)) {
    COMPAS_CUDA_CHECK(cuCtxPushCurrent(impl_->context));
}

CudaContextGuard::~CudaContextGuard() {
    try {
        CUcontext current;
        COMPAS_CUDA_CHECK(cuCtxPopCurrent(&current));
    } catch (const CudaException& e) {
        std::cerr << "ignoring cuda error: " << e.what() << "\n";
    }
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

cublasHandle_t CudaContext::cublas_handle() const {
    return impl_->cublas_handle;
}

CudaBuffer::CudaBuffer(const CudaContext& context, size_t nbytes) :
    context_(context),
    device_ptr_(CUdeviceptr {}),
    nbytes_(0) {
    if (nbytes > 0) {
        CudaContextGuard guard {context_};
        COMPAS_CUDA_CHECK(cuMemAlloc((CUdeviceptr*)&device_ptr_, nbytes));
        nbytes_ = nbytes;
    }
}

CudaBuffer::~CudaBuffer() {
    if (nbytes_ > 0) {
        try {
            CudaContextGuard guard {context_};
            COMPAS_CUDA_CHECK(cuMemFree((CUdeviceptr)device_ptr_));
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

void CudaBuffer::fill(
    const void* element_ptr,
    size_t element_nbytes,
    size_t offset,
    size_t nbytes) {
    COMPAS_ASSERT(element_nbytes > 0 && nbytes % element_nbytes == 0);

    bool all_equal = true;
    for (size_t i = 1; i < element_nbytes; i++) {
        if (static_cast<const char*>(element_ptr)[i] != static_cast<const char*>(element_ptr)[0]) {
            all_equal = false;
        }
    }

    CudaContextGuard guard {context_};
    CUdeviceptr ptr = device_ptr_ + offset;

    if (all_equal || element_nbytes == 1) {
        char value = static_cast<const char*>(element_ptr)[0];
        COMPAS_CUDA_CHECK(cuMemsetD8(ptr, value, nbytes));
    } else if (element_nbytes == 2) {
        uint16_t value = static_cast<const uint16_t*>(element_ptr)[0];
        COMPAS_CUDA_CHECK(cuMemsetD16(ptr, value, nbytes));
    } else if (element_nbytes == 4) {
        uint32_t value = static_cast<const uint32_t*>(element_ptr)[0];
        COMPAS_CUDA_CHECK(cuMemsetD32(ptr, value, nbytes));
    } else {
        COMPAS_PANIC("fill can only be performed using 8, 16, or 32 bit values");
    }
}

}  // namespace compas