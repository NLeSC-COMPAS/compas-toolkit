#pragma once

#include "compas/core/backends.h"
#include "compas/core/complex_type.h"
#include "compas/core/view.h"

namespace compas {

/**
 * The fields that are stored in `TissueParameters` represented as integers. For example, `TissueParameterField::T1`
 * is `0`, thus the first row of `TissueParameters` are the T1 values of the voxels.
 */
namespace TissueParameterField {
enum {
    T1 = 0,
    T2 = 1,
    B1 = 2,
    B0 = 3,
    RHO_X = 4,
    RHO_Y = 5,
    X = 6,
    Y = 7,
    Z = 8,
    NUM_FIELDS = 9,
};
}

struct TissueVoxel {
    float T1;
    float T2;
    float B1;
    float B0;
    cfloat rho;
    float x;
    float y;
    float z;
};

/**
 * Device-side representation of `TissueParameters`
 */
struct TissueParametersView {
    TissueParametersView(GPUSubview<float, 2> parameters = {}) : parameters(parameters) {}

    GPUSubview<float, 2> parameters;
    bool has_z = true;
    bool has_b0 = true;
    bool has_b1 = true;

    /**
     * Returns the parameters for the voxel at location `i`.
     */
    COMPAS_DEVICE TissueVoxel get(index_t i) const {
        TissueVoxel voxel;

        voxel.T1 = parameters[TissueParameterField::T1][i];
        voxel.T2 = parameters[TissueParameterField::T2][i];

        voxel.B1 = has_b1 ? parameters[TissueParameterField::B1][i] : 1;
        voxel.B0 = has_b0 ? parameters[TissueParameterField::B0][i] : 0;

        voxel.rho = {
            parameters[TissueParameterField::RHO_X][i],
            parameters[TissueParameterField::RHO_Y][i]};

        voxel.x = parameters[TissueParameterField::X][i];
        voxel.y = parameters[TissueParameterField::Y][i];
        voxel.z = has_z ? parameters[TissueParameterField::Z][i] : 0;

        return voxel;
    }
};

}  // namespace compas