#include "compas/parameters/tissue.h"
#include "compas/sequences/pssfp.h"
#include "compas/sequences/pssfp_view.h"

namespace compas {

/**
 * Simulate the magnetization at echo times for the tissue parameters given by `parameters` for the given pSSFP sequence
 * parameters.
 *
 * @param context The Context.
 * @param parameters The tissue parameters.
 * @param sequence The pSSFP sequence.
 * @return An array of dimensions `[nreadouts, nvoxels]` giving the magnetization at each voxel for each readout at
 *         echo times.
 */
Array<cfloat, 2> simulate_magnetization(
    const CompasContext& context,
    TissueParameters parameters,
    pSSFPSequence sequence);
}  // namespace compas