#pragma once

#include "core/context.h"
#include "parameters/tissue.h"
#include "trajectories/spiral_view.cuh"
#include "trajectories/trajectory.h"

namespace compas {

enum struct SimulateSignalMethod {
    Direct,
    MatmulPedantic,
    Matmul,
    MatmulTF32,
    MatmulBF16,
};

void magnetization_to_signal(
    const CudaContext& context,
    CudaArray<cfloat, 3>& signal,
    CudaArray<cfloat, 2> echos,
    TissueParameters parameters,
    Trajectory trajectory,
    CudaArray<float, 2> coil_sensitivities,
    SimulateSignalMethod method = SimulateSignalMethod::Direct);

}  // namespace compas