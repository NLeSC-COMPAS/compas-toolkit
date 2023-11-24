#include <iostream>

#include "core/context.h"
#include "core/utils.h"
#include "simulate/signal.h"
#include "simulate/signal_kernels.cuh"

namespace compas {

void simulate_signal_cartesian(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities) {
    CudaContextGuard guard {context};

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

    auto exponents = context.allocate<cfloat, 2>({samples_per_readout, nvoxels});
    auto factors = context.allocate<cfloat>(echos.shape());

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_signal_factors<<<grid_dim, block_dim>>>(
        factors.view_mut(),
        echos,
        parameters,
        trajectory);
    COMPAS_CUDA_CHECK(cudaGetLastError());

    block_dim = {256};
    grid_dim = {div_ceil(uint(nvoxels), block_dim.x)};

    kernels::prepare_signal_cartesian<<<grid_dim, block_dim>>>(
        exponents.view_mut(),
        parameters,
        trajectory);
    COMPAS_CUDA_CHECK(cudaGetLastError());

    const uint block_size_x = 64;
    const uint block_size_y = 1;
    const uint threads_cooperative = 32;
    const uint samples_per_thread = 8;
    const uint readouts_per_thread = 1;
    const uint coils_per_thread = 4;

    block_dim = {block_size_x, block_size_y};
    grid_dim = {
        div_ceil(
            div_ceil(uint(samples_per_readout), samples_per_thread) * threads_cooperative,
            block_size_x),
        div_ceil(uint(nreadouts), readouts_per_thread * block_size_y),
        div_ceil(uint(ncoils), uint(coils_per_thread)),
    };

    kernels::sum_signal_cartesian<
        block_size_x * block_size_y,
        threads_cooperative,
        samples_per_thread,
        readouts_per_thread,
        coils_per_thread>
        <<<grid_dim, block_dim>>>(signal, exponents.view(), factors.view(), coil_sensitivities);

    COMPAS_CUDA_CHECK(cudaGetLastError());
}

void simulate_signal_spiral(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    SpiralTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities) {
    CudaContextGuard guard {context};

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

    auto factors = context.allocate<cfloat>(echos.shape());
    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_signal_factors<<<grid_dim, block_dim>>>(
        factors.view_mut(),
        echos,
        parameters,
        trajectory);
    COMPAS_CUDA_CHECK(cudaGetLastError());

    auto exponents = context.allocate<cfloat>(echos.shape());
    block_dim = {32, 4};
    grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_signal_spiral<<<grid_dim, block_dim>>>(
        exponents.view_mut(),
        parameters,
        trajectory);
    COMPAS_CUDA_CHECK(cudaGetLastError());

    const uint threads_per_block = 64;
    const uint threads_cooperative = 32;
    const uint samples_per_thread = 8;
    const uint coils_per_thread = 1;

    block_dim = {threads_per_block};
    grid_dim = {
        div_ceil(
            div_ceil(uint(samples_per_readout), samples_per_thread) * threads_cooperative,
            threads_per_block),
        uint(nreadouts),
        div_ceil(uint(ncoils), uint(coils_per_thread)),
    };

    kernels::sum_signal_spiral<
        threads_per_block,
        threads_cooperative,
        samples_per_thread,
        coils_per_thread>
        <<<grid_dim, block_dim>>>(signal, exponents.view(), factors.view(), coil_sensitivities);

    COMPAS_CUDA_CHECK(cudaGetLastError());
}

void simulate_signal(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    Trajectory trajectory,
    cuda_view<float, 2> coil_sensitivities) {
    if (const auto c = trajectory.as_cartesian()) {
        simulate_signal_cartesian(
            context,
            signal,
            echos,
            parameters,
            c->view(),
            coil_sensitivities);
    } else if (const auto s = trajectory.as_spiral()) {
        simulate_signal_spiral(  //
            context,
            signal,
            echos,
            parameters,
            s->view(),
            coil_sensitivities);
    } else {
        COMPAS_PANIC("invalid trajectory type");
    }
}
}  // namespace compas
