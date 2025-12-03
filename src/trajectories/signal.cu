#include <iostream>

#include "compas/core/context.h"
#include "compas/core/utils.h"
#include "compas/trajectories/cartesian.h"
#include "compas/trajectories/signal.h"
#include "compas/trajectories/spiral.h"
#include "compas/utils/gemm.h"
#include "signal_kernels.cuh"

namespace compas {

void magnetization_to_signal_cartesian_direct(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,
    GPUSubview<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubviewMut<cfloat, 2> sample_decay,
    GPUSubviewMut<cfloat, 2> readout_echos) {
    int voxel_begin = voxels.begin;
    int voxel_end = voxels.end;
    int nvoxels = voxels.size();
    int ncoils = kmm::checked_cast<int>(coil_sensitivities.size(0));
    int nreadouts = trajectory.nreadouts;
    int samples_per_readout = trajectory.samples_per_readout;

    COMPAS_CHECK(coil_sensitivities.begin(1) <= voxel_begin);
    COMPAS_CHECK(coil_sensitivities.end(1) >= voxel_end);

    COMPAS_CHECK(signal.size(0) == ncoils);
    COMPAS_CHECK(signal.size(1) == nreadouts);
    COMPAS_CHECK(signal.size(2) == samples_per_readout);

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.begin(1) <= voxel_begin);
    COMPAS_CHECK(echos.end(1) >= voxel_end);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_readout_echos<<<grid_dim, block_dim, 0, context>>>(
        voxels,
        nreadouts,
        readout_echos,
        echos,
        parameters,
        trajectory);
    COMPAS_GPU_CHECK(gpuGetLastError());

    block_dim = {256};
    grid_dim = {div_ceil(uint(nvoxels), block_dim.x)};

    kernels::prepare_sample_decay_cartesian<<<grid_dim, block_dim, 0, context>>>(
        voxels,
        samples_per_readout,
        sample_decay,
        parameters,
        trajectory);
    COMPAS_GPU_CHECK(gpuGetLastError());

    // TODO: Are these faster?
    //    const uint threads_cooperative = 32;
    //    const uint samples_per_thread = 8;
    //    const uint readouts_per_thread = 1;
    //    const uint coils_per_thread = 4;

    const uint block_size_x = 64;
    const uint block_size_y = 4;
    const uint threads_cooperative = 16;
    const uint samples_per_thread = 4;
    const uint readouts_per_thread = 6;
    const uint coils_per_thread = 1;

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
        coils_per_thread><<<grid_dim, block_dim, 0, context>>>(
        voxel_begin,
        nvoxels,
        ncoils,
        nreadouts,
        samples_per_readout,
        signal.data_at(0, 0, 0),
        sample_decay.data_at(0, voxel_begin),
        readout_echos.data_at(0, voxel_begin),
        coil_sensitivities.data_at(0, voxel_begin));

    COMPAS_GPU_CHECK(gpuGetLastError());
    context.synchronize();
}

template<typename ComputeT = float>
void magnetization_to_signal_cartesian_gemm(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,
    GPUView<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    GPUView<cfloat, 2> coil_sensitivities,
    GPUViewMut<float, 3> temp_signal,
    GPUViewMut<ComputeT, 3> exponents,
    GPUViewMut<ComputeT, 3> factors,
    GemmComputeMethod compute_type) {
    int ncoils = kmm::checked_cast<int>(coil_sensitivities.size(0));
    int nreadouts = trajectory.nreadouts;
    int nvoxels = voxels.size();
    int samples_per_readout = trajectory.samples_per_readout;

    COMPAS_CHECK(coil_sensitivities.size(1) == voxels.size());

    COMPAS_CHECK(signal.size(0) == ncoils);
    COMPAS_CHECK(signal.size(1) == nreadouts);
    COMPAS_CHECK(signal.size(2) == samples_per_readout);

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.size(1) == nvoxels);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_readout_echos_planar<ComputeT><<<grid_dim, block_dim, 0, context.stream()>>>(
        voxels,
        nreadouts,
        factors,
        echos,
        parameters,
        trajectory);
    COMPAS_GPU_CHECK(gpuGetLastError());

    for (index_t icoil = 0; icoil < ncoils; icoil++) {
        block_dim = {256};
        grid_dim = {div_ceil(uint(nvoxels), block_dim.x)};

        kernels::
            prepare_sample_decay_cartesian_with_coil<<<grid_dim, block_dim, 0, context.stream()>>>(
                voxels,
                samples_per_readout,
                exponents,
                coil_sensitivities.drop_axis<0>(icoil),
                parameters,
                trajectory);
        COMPAS_GPU_CHECK(gpuGetLastError());

        // temp_signal = factors * exponents
        compute_complex_gemm(context, temp_signal, factors, exponents, 1.0F, 0.0F, compute_type);

        // signal[icoil] = temp_signal
        convert_planar_to_complex(context, signal.drop_axis(icoil), temp_signal);
    }

    COMPAS_GPU_CHECK(gpuGetLastError());
}

void magnetization_to_signal_spiral(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,
    GPUView<cfloat, 2> echos,
    TissueParametersView parameters,
    SpiralTrajectoryView trajectory,
    GPUView<cfloat, 2> coil_sensitivities,
    GPUViewMut<cfloat, 2> sample_decay,
    GPUViewMut<cfloat, 2> readout_echos) {
    int ncoils = kmm::checked_cast<int>(coil_sensitivities.size(0));
    int nreadouts = trajectory.nreadouts;
    int samples_per_readout = trajectory.samples_per_readout;

    COMPAS_CHECK(coil_sensitivities.size(1) == voxels.size());

    COMPAS_CHECK(signal.size(0) == ncoils);
    COMPAS_CHECK(signal.size(1) == nreadouts);
    COMPAS_CHECK(signal.size(2) == samples_per_readout);

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.size(1) == voxels.size());

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {
        div_ceil(uint(voxels.size()), block_dim.x),
        div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_readout_echos<<<grid_dim, block_dim, 0, context.stream()>>>(
        voxels,
        nreadouts,
        readout_echos,
        echos,
        parameters,
        trajectory);
    COMPAS_GPU_CHECK(gpuGetLastError());

    block_dim = {32, 4};
    grid_dim = {div_ceil(uint(voxels.size()), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_sample_decay_spiral<<<grid_dim, block_dim, 0, context.stream()>>>(
        sample_decay,
        parameters,
        trajectory);
    COMPAS_GPU_CHECK(gpuGetLastError());

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
        coils_per_thread><<<grid_dim, block_dim, 0, context.stream()>>>(
        signal,
        sample_decay,
        readout_echos,
        coil_sensitivities);

    COMPAS_GPU_CHECK(gpuGetLastError());
}

GemmComputeMethod cublas_compute_type_from_simulate_method(SimulateSignalMethod method) {
    switch (method) {
        case SimulateSignalMethod::MatmulPedantic:
            return GemmComputeMethod::Pedantic;
        case SimulateSignalMethod::Matmul:
            return GemmComputeMethod::Regular;
        case SimulateSignalMethod::MatmulFast:
            return GemmComputeMethod::Fast;
        default:
            COMPAS_ERROR("invalid value for `SimulateSignalMethod`");
    }
}

Array<cfloat, 3> magnetization_to_signal(
    const CompasContext& context,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    const Trajectory& trajectory,
    Array<cfloat, 2> coil_sensitivities,
    SimulateSignalMethod method) {
    using namespace kmm::placeholders;

    int ncoils = kmm::checked_cast<int>(coil_sensitivities.size(0));
    int nvoxels = parameters.nvoxels;
    int chunk_size = parameters.chunk_size;
    int nreadouts = trajectory.nreadouts;
    int samples_per_readout = trajectory.samples_per_readout;

    auto signal = Array<cfloat, 3> {{ncoils, nreadouts, samples_per_readout}};

    if (const auto* cart = dynamic_cast<const CartesianTrajectory*>(&trajectory)) {
        if (method == SimulateSignalMethod::Naive) {
            context.submit_kernel(
                {uint(samples_per_readout), uint(nreadouts), uint(ncoils)},
                256,
                kernels::sum_signal_cartesian_naive,
                nvoxels,
                write(signal),
                echos,
                parameters,
                *cart,
                coil_sensitivities);

        } else if (method == SimulateSignalMethod::Direct) {
            auto temp_exponents = Array<cfloat, 2> {{samples_per_readout, nvoxels}};
            auto temp_factors = Array<cfloat, 2> {{nreadouts, nvoxels}};

            context.parallel_device(
                {nvoxels, nreadouts},
                {chunk_size, nreadouts},
                magnetization_to_signal_cartesian_direct,
                _x,
                reduce(kmm::Reduction::Sum, signal),
                echos[_y][_x],
                parameters.data[_][_x],
                *cart,
                coil_sensitivities[_][_x],
                write(temp_exponents[_][_x]),
                write(temp_factors[_y][_x]));
        } else if (method == SimulateSignalMethod::MatmulLow) {
            auto temp_exponents =
                Array<kernel_float::bfloat16_t, 3> {{2, samples_per_readout, nvoxels}};
            auto temp_factors = Array<kernel_float::bfloat16_t, 3> {{2, nreadouts, nvoxels}};
            auto temp_signal = Array<float, 3> {{2, nreadouts, samples_per_readout}};

            context.submit_device(
                magnetization_to_signal_cartesian_gemm<kernel_float::bfloat16_t>,
                nvoxels,
                write(signal),
                echos,
                parameters,
                *cart,
                coil_sensitivities,
                write(temp_signal),
                write(temp_exponents),
                write(temp_factors),
                GemmComputeMethod::Fast);
        } else {
            auto temp_exponents = Array<float, 3> {{2, samples_per_readout, nvoxels}};
            auto temp_factors = Array<float, 3> {{2, nreadouts, nvoxels}};
            auto temp_signal = Array<float, 3> {{2, nreadouts, samples_per_readout}};

            context.submit_device(
                magnetization_to_signal_cartesian_gemm<float>,
                nvoxels,
                write(signal),
                echos,
                parameters,
                *cart,
                coil_sensitivities,
                write(temp_signal),
                write(temp_exponents),
                write(temp_factors),
                cublas_compute_type_from_simulate_method(method));
        }
    } else if (const auto* s = dynamic_cast<const SpiralTrajectory*>(&trajectory)) {
        auto temp_exponents = Array<cfloat, 2>(echos.size());
        auto temp_factors = Array<cfloat, 2>(echos.size());

        context.submit_device(
            magnetization_to_signal_spiral,
            _x,
            write(signal),
            echos,
            parameters,
            *s,
            coil_sensitivities,
            write(temp_exponents),
            write(temp_factors));
    } else {
        COMPAS_ERROR("invalid trajectory type");
    }

    return signal;
}
}  // namespace compas
