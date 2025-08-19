#pragma once

#include "backends.h"
#include "macros.h"

#if COMPAS_IS_DEVICE
namespace compas {

[[noreturn]] __device__ __noinline__ void
raise_error(const char* message, const char* file, const int line, const char* func) {
    printf(
        "[GPU thread (%d, %d, %d)(%d, %d, %d)] COMPAS error: (%s:%d:%s): %s\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        file,
        line,
        func,
        message);

    asm("trap;");
    while (1)
        ;
}
}  // namespace compas

#else
    #include <sstream>
    #include <stdexcept>

namespace compas {
struct CompasException: std::runtime_error {
    CompasException(std::string message) : std::runtime_error(std::move(message)) {}
};

[[noreturn]] inline void
raise_error(const char* message, const char* file, const int line, const char* func) {
    std::stringstream ss;
    ss << "COMPAS error occurred (" << file << ":" << line << ":" << func << "): " << message;
    throw CompasException(ss.str());
}
}  // namespace compas
#endif

#define COMPAS_ERROR(message)                                         \
    do {                                                              \
        ::compas::raise_error(message, __FILE__, __LINE__, __func__); \
        while (1) {                                                   \
        }                                                             \
    } while (0)

#define COMPAS_CHECK(...)                                    \
    do {                                                     \
        if (!(__VA_ARGS__)) {                                \
            COMPAS_ERROR("assertion failed: " #__VA_ARGS__); \
        }                                                    \
    } while (0)

#ifdef NDEBUG
    #define COMPAS_DEBUG_ASSERT(...) COMPAS_ASSUME((__VA_ARGS__))
#else
    #define COMPAS_DEBUG_ASSERT(...) COMPAS_CHECK((__VA_ARGS__))
#endif