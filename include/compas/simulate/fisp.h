#include "compas/parameters/tissue.h"
#include "compas/parameters/tissue_view.cuh"
#include "compas/sequences/fisp.h"

namespace compas {

/**
 * Simulate the magnetization at echo times for the tissue parameters given by `parameters` for the given FISP sequence
 * parameters.
 *
 * @param context The Context.
 * @param parameters The tissue parameters.
 * @param sequence The FISP sequence.
 * @return An array of dimensions `[nreadouts, nvoxels]` giving the magnetization at each voxel for each readout at
 *         echo times.
 */
Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    FISPSequence sequence);

}  // namespace compas