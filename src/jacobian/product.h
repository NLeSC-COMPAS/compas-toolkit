#include "core/complex_type.h"
#include "core/context.h"
#include "parameters/tissue_view.cuh"
#include "trajectories/cartesian_view.cuh"

namespace compas {

void compute_jacobian(
    const CudaContext& ctx,
    cuda_view_mut<cfloat, 2> Jv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 2> vector);

void compute_jacobian_transposed(
    const CudaContext& ctx,
    cuda_view_mut<cfloat, 2> JHv,
    cuda_view<cfloat, 2> echos,
    cuda_view<cfloat, 3> delta_echos,
    TissueParametersView parameters,
    CartesianTrajectoryView trajectory,
    cuda_view<float, 2> coil_sensitivities,
    cuda_view<cfloat, 2> vector);

}  // namespace compas