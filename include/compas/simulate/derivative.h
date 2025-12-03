#include "compas/parameters/tissue.h"
#include "compas/sequences/fisp.h"
#include "compas/sequences/pssfp.h"

namespace compas {

static constexpr float DEFAULT_FINITE_DIFFERENCE_DELTA = 1e-4F;

/**
 * Calculate the derivative of the magnetization at echo times given by `simulate_magnetization`.
 *
 * @param context The context.
 * @param field One of the fields given by `TissueParameterField`.
 * @param echos The echos calculated by `simulate_magnetization`.
 * @param parameters The tissue parameters.
 * @param sequence The pSSFP sequence
 * @param delta The Δ used for calculating the finite difference method.
 * @return An array of dimensions `[nreadouts, nvoxels]` giving the derivative over the given field of the magnetization
 *         at each voxel for each readout.
 */
Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence,
    float delta = DEFAULT_FINITE_DIFFERENCE_DELTA);

/**
 * Calculate the derivative of the magnetization at echo times given by `simulate_magnetization`.
 *
 * @param context The context.
 * @param field One of the fields given by `TissueParameterField`.
 * @param echos The echos calculated by `simulate_magnetization`.
 * @param parameters The tissue parameters.
 * @param sequence The FISP sequence
 * @param delta The Δ used for calculating the finite difference method.
 * @return An array of dimensions `[nreadouts, nvoxels]` giving the derivative over the given field of the magnetization
 *         at each voxel for each readout.
 */
Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    FISPSequence sequence,
    float delta = DEFAULT_FINITE_DIFFERENCE_DELTA);

}  // namespace compas