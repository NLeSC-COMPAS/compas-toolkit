#pragma once

#include "core/view.h"
#include "parameters/tissue_kernels.cuh"

namespace compas {

struct Isochromat: vfloat3 {
    COMPAS_HOST_DEVICE Isochromat() : vfloat3(0.0f, 0.0f, 1.0f) {}
    COMPAS_HOST_DEVICE Isochromat(float x, float y, float z) : vfloat3(x, y, z) {}
    COMPAS_HOST_DEVICE Isochromat(vfloat3 v) : vfloat3(v) {}

    COMPAS_HOST_DEVICE
    Isochromat
    rotate(cfloat gamma_dt_RF, float gamma_dt_GR, float r, float dt, const TissueVoxel& p) const {
        Isochromat m = *this;

        // Determine rotation vector a
        auto ax = gamma_dt_RF.real();
        auto ay = -gamma_dt_RF.imag();
        auto az = -(gamma_dt_GR * r);

        ax *= p.B1;
        ay *= p.B1;
        az -= dt * float(2 * M_PI) * p.B0;

        vfloat3 a = {ax, ay, az};

        // Angle of rotation is norm of rotation vector
        auto theta = norm(a);

        if (fabsf(theta) < 1e-9f) {  // theta != 0
            // Normalize rotation vector
#if COMPAS_IS_DEVICE
            vfloat3 k = a * __fdividef(1.0f, theta);
#else
            vfloat3 k = a / theta;
#endif

            float sin_theta, cos_theta;
#if COMPAS_IS_DEVICE
            __sincosf(theta, &sin_theta, &cos_theta);
#else
            sin_theta = sinf(theta);
            cos_theta = cosf(theta);
#endif

            // Perform rotation (Rodrigues formula)
            m = (cos_theta * m) + (sin_theta * cross(k, m))
                + (dot(k, m) * (float(1) - cos_theta) * k);
        }

        return m;
    }

    COMPAS_HOST_DEVICE
    Isochromat rotate(vfloat3 gamma_dt_GR, vfloat3 r, float dt, const TissueVoxel& p) const {
        // Determine rotation vector a
        auto az = -dot(gamma_dt_GR, r);
        az -= dt * float(2 * M_PI) * p.B0;

        // Angle of rotation
        auto theta = az;
        float sin_theta, cos_theta;
#if COMPAS_IS_DEVICE
        __sincosf(theta, &sin_theta, &cos_theta);
#else
        sin_theta = sinf(theta);
        cos_theta = cosf(theta);
#endif

        // Perform rotation in XY plane
        Isochromat m = *this;
        m.x = cos_theta * m.x - sin_theta * m.y;
        m.y = cos_theta * m.y + sin_theta * m.y;

        return m;
    }

    COMPAS_HOST_DEVICE
    Isochromat decay(float E1, float E2) const {
        return {x * E2, y * E2, z * E1};
    }

    COMPAS_HOST_DEVICE
    Isochromat regrowth(float E1) const {
        return {x, y, z + (1 - E1)};
    }

    COMPAS_HOST_DEVICE
    Isochromat invert(const TissueVoxel& p) const {
        float theta = float(M_PI);
        theta *= p.B1;

        float cos_theta;
#if COMPAS_IS_DEVICE
        cos_theta = __cosf(theta);
#else
        cos_theta = cosf(theta);
#endif

        return {x, y, cos_theta * z};
    }

    COMPAS_HOST_DEVICE
    Isochromat invert() const {
        return {x, y, -z};
    }
};

};  // namespace compas