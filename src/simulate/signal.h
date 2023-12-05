#pragma once

#include "core/context.h"
#include "parameters/tissue.h"
#include "trajectories/multi.h"
#include "trajectories/spiral_view.cuh"

namespace compas {

void simulate_signal(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    const Trajectory& trajectory,
    cuda_view<float, 2> coil_sensitivities);

enum struct SimulateSignalMethod {
    Direct,
    MatmulPedantic,
    Matmul,
    MatmulTF32,
    MatmulBF16,
};

void simulate_signal_cartesian(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    SimulateSignalMethod method = SimulateSignalMethod::Direct);

void simulate_signal_spiral(
    const CudaContext& context,
    cuda_view_mut<cfloat, 3> signal,
    cuda_view<cfloat, 2> echos,
    TissueParametersView parameters,
    SpiralTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities);

}  // namespace compas