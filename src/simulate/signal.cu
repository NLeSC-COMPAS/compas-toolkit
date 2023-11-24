#include <iostream>

#include "core/context.h"
#include "core/utils.h"
#include "simulate/signal.h"
#include "simulate/signal_kernels.cuh"

namespace compas {

void simulate_signal_cartesian_direct(
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

void simulate_signal_cartesian_gemm(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cublasComputeType_t compute_type) {
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

    for (index_t icoil = 0; icoil < ncoils; icoil++) {
        block_dim = {256};
        grid_dim = {div_ceil(uint(nvoxels), block_dim.x)};

        kernels::prepare_signal_cartesian_with_coil<<<grid_dim, block_dim>>>(
            exponents.view_mut(),
            coil_sensitivities.drop_leading_axis(icoil),
            parameters,
            trajectory);
        COMPAS_CUDA_CHECK(cudaGetLastError());

        cuComplex alpha = {1, 0};
        cuComplex beta = {0, 0};

        cudaDataType_t output_type = CUDA_C_32F;
        cudaDataType_t input_type = CUDA_C_32F;
        cublasGemmAlgo_t compute_algo = CUBLAS_GEMM_DEFAULT;

        COMPAS_CUDA_CHECK(cublasSetStream(context.cublas_handle(), nullptr));
        COMPAS_CUDA_CHECK(cublasGemmEx(
            context.cublas_handle(),
            CUBLAS_OP_T,  // transa
            CUBLAS_OP_N,  // transb
            samples_per_readout,  // m
            nreadouts,  // n
            nvoxels,  // k
            &alpha,  // alpha
            exponents.device_data(),  // A
            input_type,  // A type
            nvoxels,  // lda
            factors.device_data(),  // B
            input_type,  // B type
            nvoxels,  // ldb
            &beta,  //beta
            signal.data() + signal.stride(0) * icoil,  // C
            output_type,  // C type
            samples_per_readout,  // ldc
            compute_type,
            compute_algo));
    }

    COMPAS_CUDA_CHECK(cudaGetLastError());
}

void simulate_signal_cartesian(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    SimulateSignalMethod method) {
    if (method == SimulateSignalMethod::Direct) {
        simulate_signal_cartesian_direct(
            context,
            signal,
            echos,
            parameters,
            trajectory,
            coil_sensitivities);
    } else {
        cublasComputeType_t compute_type = [&] {
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
        }();

        simulate_signal_cartesian_gemm(
            context,
            signal,
            echos,
            parameters,
            trajectory,
            coil_sensitivities,
            compute_type);
    }
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
