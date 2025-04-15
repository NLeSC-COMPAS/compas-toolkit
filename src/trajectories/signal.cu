#include <iostream>

#include "compas/core/context.h"
#include "compas/core/utils.h"
#include "compas/trajectories/cartesian.h"
#include "compas/trajectories/signal.h"
#include "compas/trajectories/spiral.h"
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

    COMPAS_ASSERT(coil_sensitivities.begin(1) <= voxel_begin);
    COMPAS_ASSERT(coil_sensitivities.end(1) >= voxel_end);

    COMPAS_ASSERT(signal.size(0) == ncoils);
    COMPAS_ASSERT(signal.size(1) == nreadouts);
    COMPAS_ASSERT(signal.size(2) == samples_per_readout);

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.begin(1) <= voxel_begin);
    COMPAS_ASSERT(echos.end(1) >= voxel_end);

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

    const uint block_size_x = 32;
    const uint block_size_y = 8;
    const uint threads_cooperative = 32;
    const uint samples_per_thread = 1;
    const uint readouts_per_thread = 1;
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
        voxels,
        signal,
        sample_decay,
        readout_echos,
        coil_sensitivities);

    COMPAS_GPU_CHECK(gpuGetLastError());
    context.synchronize();
}

void magnetization_to_signal_cartesian_gemm(
    const kmm::DeviceResource& context,
    kmm::Range<index_t> voxels,
    GPUViewMut<cfloat, 3> signal,
    GPUView<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    GPUView<cfloat, 2> coil_sensitivities,
    GPUViewMut<cfloat, 2> exponents,
    GPUViewMut<cfloat, 2> factors,
    cublasComputeType_t compute_type) {
    int ncoils = kmm::checked_cast<int>(coil_sensitivities.size(0));
    int nreadouts = trajectory.nreadouts;
    int nvoxels = voxels.size();
    int samples_per_readout = trajectory.samples_per_readout;

    COMPAS_ASSERT(coil_sensitivities.size(1) == voxels.size());

    COMPAS_ASSERT(signal.size(0) == ncoils);
    COMPAS_ASSERT(signal.size(1) == nreadouts);
    COMPAS_ASSERT(signal.size(2) == samples_per_readout);

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    dim3 block_dim = {32, 4};
    dim3 grid_dim = {div_ceil(uint(nvoxels), block_dim.x), div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_readout_echos<<<grid_dim, block_dim, 0, context.stream()>>>(
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
                exponents,
                coil_sensitivities.drop_axis<0>(icoil),
                parameters,
                trajectory);
        COMPAS_GPU_CHECK(gpuGetLastError());

        cuComplex alpha = {1, 0};
        cuComplex beta = {0, 0};

        cudaDataType_t output_type = CUDA_C_32F;
        cudaDataType_t input_type = CUDA_C_32F;
        cublasGemmAlgo_t compute_algo = CUBLAS_GEMM_DEFAULT;

        COMPAS_GPU_CHECK(cublasSetStream(context.blas(), context.stream()));
        COMPAS_GPU_CHECK(cublasGemmEx(
            context.blas(),
            CUBLAS_OP_T,  // transa
            CUBLAS_OP_N,  // transb
            samples_per_readout,  // m
            nreadouts,  // n
            nvoxels,  // k
            &alpha,  // alpha
            exponents.data(),  // A
            input_type,  // A type
            nvoxels,  // lda
            factors.data(),  // B
            input_type,  // B type
            nvoxels,  // ldb
            &beta,  //beta
            signal.data() + signal.stride(0) * icoil,  // C
            output_type,  // C type
            samples_per_readout,  // ldc
            compute_type,
            compute_algo));
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

    COMPAS_ASSERT(coil_sensitivities.size(1) == voxels.size());

    COMPAS_ASSERT(signal.size(0) == ncoils);
    COMPAS_ASSERT(signal.size(1) == nreadouts);
    COMPAS_ASSERT(signal.size(2) == samples_per_readout);

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == voxels.size());

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

cublasComputeType_t cublas_compute_type_from_simulate_method(SimulateSignalMethod method) {
    switch (method) {
        case SimulateSignalMethod::MatmulPedantic:
            return CUBLAS_COMPUTE_32F_PEDANTIC;
        case SimulateSignalMethod::Matmul:
            return CUBLAS_COMPUTE_32F;
        case SimulateSignalMethod::MatmulBF16:
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        case SimulateSignalMethod::MatmulTF32:
            return CUBLAS_COMPUTE_32F_FAST_TF32;
        default:
            COMPAS_PANIC("invalid value for `SimulateSignalMethod`");
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
        auto temp_exponents = Array<cfloat, 2> {{samples_per_readout, nvoxels}};
        auto temp_factors = Array<cfloat, 2> {{nreadouts, nvoxels}};

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
        } else {
            context.submit_device(
                magnetization_to_signal_cartesian_gemm,
                nvoxels,
                write(signal),
                echos,
                parameters,
                *cart,
                coil_sensitivities,
                write(temp_exponents),
                write(temp_factors),
                cublas_compute_type_from_simulate_method(method));
        }
    } else if (const auto* s = dynamic_cast<const SpiralTrajectory*>(&trajectory)) {
        auto temp_exponents = Array<cfloat, 2>(echos.shape());
        auto temp_factors = Array<cfloat, 2>(echos.shape());

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
        COMPAS_PANIC("invalid trajectory type");
    }

    return signal;
}
}  // namespace compas
