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

    dim3 block_dim = {16, 16};
    dim3 grid_dim = {
        div_ceil(uint(nvoxels), block_dim.x),
        div_ceil(uint(nreadouts), block_dim.y)};

    kernels::prepare_signal<<<grid_dim, block_dim>>>(
        exponents.view_mut(),
        factors.view_mut(),
        echos.view(),
        parameters.view(),
        trajectory.view());

    for (int coil = 0; coil < ncoils; coil++) {
        block_dim = {256};
        grid_dim = {div_ceil(uint(samples_per_readout), block_dim.x)};

        kernels::sum_signal<<<grid_dim, block_dim>>>(
            signal.slice(coil).view_mut(),
            exponents.view(),
            factors.view(),
            coil_sensitivities.slice(coil).view());
    }
}

}  // namespace compas
