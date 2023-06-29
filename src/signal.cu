#include "signal.h"
#include "signal_kernels.cuh"

namespace compas {

void simulate_signal(
        CudaContext ctx,
        CudaView<float, 1> signal, // t
        CudaView<float, 2> echos, // nvoxels x nreadouts
        const TissueParameters& parameters, // nvoxels
        const Trajectory& trajectory, // ??
        CudaView<float> coil_sensitivites // nvoxels x ncoils
) {
    size_t nvoxels = echos.size(0);
    size_t nreadouts = echos.size(1);
}

}