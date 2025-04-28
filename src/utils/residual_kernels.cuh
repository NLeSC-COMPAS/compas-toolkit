#pragma once

#include "compas/core/view.h"

namespace compas {
namespace kernels {

template<int block_size>
__device__ void reduce_within_block(float* shared_inputs, float* output) {
    auto tid = index_t(threadIdx.x);

#pragma unroll
    for (int delta = block_size / 2; delta > 0; delta /= 2) {
        if (tid + delta < block_size) {
            shared_inputs[tid] += shared_inputs[tid + delta];
        }

        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_inputs[0];
    }
}

template<int block_size>
__global__ void calculate_elementwise_difference(
    kmm::Range<index_t> range,
    const cfloat* lhs,
    const cfloat* rhs,
    cfloat* output,
    GPUViewMut<float> partial_sums) {
    __shared__ float shared_partial_sums[block_size];
    auto tid = index_t(threadIdx.x);
    auto start = index_t(blockIdx.x * block_size + threadIdx.x) + range.begin;
    auto stride = index_t(gridDim.x * block_size);

    shared_partial_sums[tid] = 0;

    for (auto i = start; i < range.end; i += stride) {
        auto diff = lhs[i] - rhs[i];
        output[i] = diff;
        shared_partial_sums[tid] += diff.norm();
    }

    __syncthreads();

    reduce_within_block<block_size>(shared_partial_sums, &partial_sums[blockIdx.x]);
}

template<int block_size>
__global__ void accumulate_partial_sums(GPUView<float> partial_sums, GPUViewMut<float> result_sum) {
    __shared__ float shared_partial_sums[block_size];
    auto tid = index_t(threadIdx.x);
    shared_partial_sums[tid] = 0;

    for (auto i = tid; i < partial_sums.size(); i += block_size) {
        shared_partial_sums[tid] += partial_sums[i];
    }

    __syncthreads();

    reduce_within_block<block_size>(shared_partial_sums, result_sum.data());
}

}  // namespace kernels
}  // namespace compas
