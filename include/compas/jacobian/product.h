#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/parameters/tissue.h"
#include "compas/trajectories/cartesian.h"

namespace compas {

Array<cfloat, 3> compute_jacobian(
    const CudaContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<float, 2> coil_sensitivities,
    Array<cfloat, 2> vector);

}  // namespace compas