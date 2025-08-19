#pragma once

#include "compas/core/vector.h"
#include "compas/core/view.h"
#include "compas/parameters/tissue_view.cuh"

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
        float theta_sq = a.x * a.x + a.y * a.y + a.z * a.z;
        float theta;

#if COMPAS_IS_CUDA
        //        theta = __sqrtf(theta_sq);
        asm("sqrt.approx.f32 %0, %1;" : "=f"(theta) : "f"(theta_sq));
#elif COMPAS_IS_HIP
        // TODO: add asm implementation?
        theta = sqrtf(theta_sq);
#else
        theta = sqrtf(theta_sq);
#endif

        if (fabsf(theta) > 1e-9f) {  // theta != 0

            // Normalize rotation vector
#if COMPAS_IS_CUDA
            float theta_rcp = 0;
            //            theta_rcp = __fdividef(1.0f, theta);
            asm("rcp.approx.f32 %0, %1;" : "=f"(theta_rcp) : "f"(theta));
            vfloat3 k = a * theta_rcp;
#elif COMPAS_IS_HIP
            // TODO: add asm implementation?
            vfloat3 k = a / theta;
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
    Isochromat rotate(float gamma_dt_GR, float r, float dt, const TissueVoxel& p) const {
        // Determine rotation vector a
        auto az = -(gamma_dt_GR * r);
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
        Isochromat m;
        m.x = cos_theta * x - sin_theta * y;
        m.y = cos_theta * y + sin_theta * x;
        m.z = z;
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