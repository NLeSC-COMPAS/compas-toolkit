#pragma once

#include <cstddef>
#include <array>
#include <cuda_runtime_api.h>

namespace compas {

struct CudaContext {

};

template <typename T, size_t N=1>
struct CudaView {
    CudaView(T* ptr, std::array<size_t, N> shape_): ptr_(ptr), shape_(shape) {}

    T* data() const {
        return ptr_;
    }

    std::array<size_t, N> shape() const {
        return shape_;
    }

    size_t size(size_t axis) const {
        return axis < N ? shape_[axis] : 1;
    }

    T* ptr_;
    std::array<size_t, N> shape_;
};

struct TissueParameters {
    CudaView<float> T_1;
    CudaView<float> T_2;
    CudaView<float> B_0;
    CudaView<float> B_1;
    CudaView<float2> rho;
    CudaView<float3> xyz;
};


struct Trajectory {
    size_t nreadouts;
    size_t nsamples_per_readout;
    float delta_t;
    CudaView<float2> k_start;
    CudaView<float2> k_delta;
};

void simulate_signal(
        CudaView<float, 1> signal, // t
        CudaView<float, 2> echos, // nvoxels x nreadouts
        const TissueParameters& parameters, // nvoxels
        const Trajectory& trajectory, // ??
        CudaView<float> coil_sensitivites // nvoxels x ncoils
);

}