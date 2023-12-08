#include "product.h"

namespace compas {
    void compute_jacobian(
            const CudaContext& ctx,
            cuda_view<cfloat, 2> echos,
            cuda_view<cfloat, 3> delta_echos,
            TissueParametersView parameters,
            const CartesianTrajectoryView& trajectory,
            cuda_view<float, 2> coil_sensitivities,
            cuda_view<cfloat, 2> vector
    ) {

    }
}