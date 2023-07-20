#include "cartesian.h"

namespace compas {

CartesianTrajectory make_cartesian_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    host_view<cfloat> delta_k) {
    COMPAS_ASSERT(k_start.size() == nreadouts);
    COMPAS_ASSERT(delta_k.size() == nreadouts);

    return CartesianTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        context.allocate(delta_k)};
}
}  // namespace compas