#include "core/complex_type.h"
#include "core/context.h"
#include "parameters/tissue.h"
#include "trajectories/cartesian.h"

namespace compas {

Array<cfloat, 2> compute_jacobian(
    const CudaContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 3> delta_echos,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<float, 2> coil_sensitivities,
    Array<cfloat, 2> vector);

Array<cfloat, 2> compute_jacobian_hermitian(
    const CudaContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 3> delta_echos,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<float, 2> coil_sensitivities,
    Array<cfloat, 2> vector);

}  // namespace compas