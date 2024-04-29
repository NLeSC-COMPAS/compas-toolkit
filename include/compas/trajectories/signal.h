#pragma once

#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/spiral_view.cuh"
#include "compas/trajectories/trajectory.h"

namespace compas {

enum struct SimulateSignalMethod {
    Direct,
    MatmulPedantic,
    Matmul,
    MatmulTF32,
    MatmulBF16,
};

Array<cfloat, 3> magnetization_to_signal(
    const CudaContext& context,
    const Array<cfloat, 2>& echos,
    const TissueParameters& parameters,
    const Trajectory& trajectory,
    const Array<float, 2>& coil_sensitivities,
    SimulateSignalMethod method = SimulateSignalMethod::Direct);

}  // namespace compas