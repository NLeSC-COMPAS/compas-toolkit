#include "core/context.h"
#include "core/complex_type.h"
#include "parameters/tissue_view.cuh"
#include "trajectories/cartesian_view.cuh"

namespace compas {

    void compute_jacobian(
            const CudaContext& ctx,
            cuda_view<cfloat, 2> echos,
            cuda_view<cfloat, 3> delta_echos,
            TissueParametersView parameters,
            const CartesianTrajectoryView& trajectory,
            cuda_view<float, 2> coil_sensitivities,
            cuda_view<cfloat, 2> vector
    );

    void compute_jacobian_transposed(
            const CudaContext& ctx,
            cuda_view<cfloat, 2> echos,
            cuda_view<cfloat, 3> delta_echos,
            TissueParametersView parameters,
            const CartesianTrajectoryView& trajectory,
            cuda_view<float, 2> coil_sensitivities,
            cuda_view<cfloat, 2> vector
    );

}