#include "core/all.h"
#include "signal_kernels.cuh"

namespace compas {

void simulate_signal(
    CudaView<cfloat, 3> signal,
    CudaView<const cfloat, 2> echos,
    TissueParameters parameters,
    SpokesTrajectory trajectory,
    CudaView<const float, 2> coil_sensitivities) {
    CudaView<cfloat, 2> exponents;
    CudaView<cfloat, 2> factors;

    simulate_signal_prepare_kernel<<<1, 1>>>(
        exponents,
        factors,
        echos,
        parameters,
        trajectory);

    simulate_signal_sum_kernel<<<1, 1>>>(
        signal,
        exponents,
        factors,
        coil_sensitivities);
}

}  // namespace compas