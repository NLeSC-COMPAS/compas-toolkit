#include "spiral.h"

namespace compas {

SpiralTrajectory make_spiral_trajectory(
    const CudaContext& context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    host_view<cfloat> k_start,
    host_view<cfloat> delta_k) {
    COMPAS_ASSERT(k_start.size() == nreadouts);
    COMPAS_ASSERT(delta_k.size() == nreadouts);

    return SpiralTrajectory {
        nreadouts,
        samples_per_readout,
        delta_t,
        context.allocate(k_start),
        context.allocate(delta_k)};
}
}  // namespace compas