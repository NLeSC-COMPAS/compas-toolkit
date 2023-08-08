#include <iostream>

#include "core/context.h"
#include "core/utils.h"
#include "parameters/tissue.h"
#include "simulate/signal.h"
#include "simulate/signal_kernels.cuh"
#include "trajectories/cartesian.h"

namespace compas {

void simulate_signal(
    const CudaContext& context,
    CudaArray<cfloat, 3> signal,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    CudaArray<float, 2> coil_sensitivities) {
    int ncoils = coil_sensitivities.size(0);
    int nvoxels = parameters.nvoxels;
    int nreadouts = trajectory.nreadouts;
    int samples_per_readout = trajectory.samples_per_readout;

    COMPAS_ASSERT(coil_sensitivities.size(1) == nvoxels);

    COMPAS_ASSERT(signal.size(0) == ncoils);
    COMPAS_ASSERT(signal.size(1) == nreadouts);
    COMPAS_ASSERT(signal.size(2) == samples_per_readout);

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    auto exponents = context.allocate<cfloat>(echos.shape());
    auto factors = context.allocate<cfloat>(echos.shape());

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_signal<<<grid_dim, block_dim>>>(
        exponents.view_mut(),
        factors.view_mut(),
        echos.view(),
        parameters.view(),
        trajectory.view());
    COMPAS_CUDA_CHECK(cudaGetLastError());

    const uint threads_per_block = 64;
    const uint threads_cooperative = 32;
    const uint samples_per_thread = 8;
    const uint coils_per_thread = 4;

    block_dim = {threads_cooperative, threads_per_block / threads_cooperative};
    grid_dim = {
        div_ceil(
            div_ceil(uint(samples_per_readout), samples_per_thread) * threads_cooperative,
            threads_per_block),
        uint(nreadouts),
        div_ceil(uint(ncoils), uint(coils_per_thread)),
    };

    kernels::
        sum_signal<threads_per_block, threads_cooperative, samples_per_thread, coils_per_thread>
        <<<grid_dim, block_dim>>>(
            signal.view_mut(),
            exponents.view(),
            factors.view(),
            coil_sensitivities.view());

    COMPAS_CUDA_CHECK(cudaGetLastError());
}
}  // namespace compas
