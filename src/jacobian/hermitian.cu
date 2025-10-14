#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/core/vector.h"
#include "compas/jacobian/hermitian.h"
#include "compas/utils/gemm.h"
#include "hermitian_kernels.cuh"

namespace compas {

void launch_jacobian_hermitian_kernel(
    kmm::DeviceResource& ctx,
    kmm::Bounds<3> subrange,
    GPUSubviewMut<cfloat, 2> JHv,
    GPUSubview<cfloat, 2> echos,
    GPUSubview<cfloat, 2> delta_echos_T1,
    GPUSubview<cfloat, 2> delta_echos_T2,
    TissueParametersView parameters,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubview<cfloat, 3> vector,
    GPUSubview<cfloat, 2> E,
    GPUSubview<cfloat, 2> dEdT2) {
    uint ncoils = coil_sensitivities.size(0);
    uint nvoxels = subrange.x.size();
    uint ns = subrange.y.size();
    uint nreadouts = subrange.z.size();

#define COMPAS_COMPUTE_JACOBIAN_IMPL(C, V, R, S, BX, BY, BZ)        \
    if (ncoils == (C) && nreadouts % (R) == 0 && ns % (S) == 0) {   \
        dim3 block_size = {BX, BY, BZ};                             \
        dim3 grid_size = {div_ceil(nvoxels, uint(V))};              \
                                                                    \
        kernels::jacobian_hermitian_product<V, R, S, C, BX, BY, BZ> \
            <<<grid_size, block_size, 0, ctx>>>(                    \
                subrange,                                           \
                JHv,                                                \
                echos,                                              \
                delta_echos_T1,                                     \
                delta_echos_T2,                                     \
                parameters,                                         \
                coil_sensitivities,                                 \
                vector,                                             \
                E,                                                  \
                dEdT2);                                             \
        return;                                                     \
    }

#define COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(C)        \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 32, 32, 2, 2) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 16, 32, 2, 2) \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 8, 8, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 4, 4, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 32, 2, 2, 32, 2, 1)  \
    COMPAS_COMPUTE_JACOBIAN_IMPL(C, 128, 1, 1, 128, 1, 1)

    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(1)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(3)
    COMPAS_COMPUTE_JACOBIAN_PER_COILS_IMPL(4)

    throw std::runtime_error("cannot support more than 4 coils");
}

Array<cfloat, 2> compute_jacobian_hermitian_naive(
    const CompasContext& ctx,
    int nreadouts,
    int ns,
    int nvoxels,
    int ncoils,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    int chunk_size = parameters.chunk_size;

    // four reconstruction parameters: T1, T2, rho_x, rho_y
    auto JHv = Array<cfloat, 2> {{4, nvoxels}};

    ctx.parallel_submit(
        {nvoxels},
        {chunk_size},
        kmm::GPUKernel(kernels::jacobian_hermitian_product_naive, 256),
        kmm::placeholders::_x,
        nreadouts,
        ns,
        ncoils,
        write(JHv),
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector);

    return JHv;
}

Array<cfloat, 2> compute_jacobian_hermitian_direct(
    const CompasContext& ctx,
    int nreadouts,
    int ns,
    int nvoxels,
    int ncoils,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    using namespace kmm::placeholders;
    int chunk_size = parameters.chunk_size;

    // four reconstruction parameters: T1, T2, rho_x, rho_y
    auto JHv = Array<cfloat, 2> {{4, nvoxels}};
    auto E = Array<cfloat, 2> {{ns, nvoxels}};
    auto dEdT2 = Array<cfloat, 2> {{ns, nvoxels}};

    auto _voxel = kmm::Axis(0);
    dim3 block_dim = {64, 4};

    ctx.parallel_submit(
        {nvoxels, ns},
        {chunk_size, ns},
        kmm::GPUKernel(kernels::compute_sample_decay, block_dim),
        _xy,
        write(E[_][_voxel]),
        write(dEdT2[_][_voxel]),
        trajectory,
        read(parameters, _voxel));

    ctx.parallel_submit(
        {nvoxels, ns, nreadouts},
        {chunk_size, ns, nreadouts},
        kmm::GPU(launch_jacobian_hermitian_kernel),
        _xyz,
        write(JHv[_][_voxel]),
        echos[_][_voxel],
        delta_echos_T1[_][_voxel],
        delta_echos_T2[_][_voxel],
        read(parameters, _voxel),
        coil_sensitivities[_][_voxel],
        vector,
        E[_][_voxel],
        dEdT2[_][_voxel]);

    return JHv;
}

template<typename ComputeT = float>
Array<cfloat, 2> compute_jacobian_hermitian_gemm(
    const CompasContext& ctx,
    GemmComputeMethod gemm,
    int nreadouts,
    int ns,
    int nvoxels,
    int ncoils,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector) {
    using namespace kmm::placeholders;
    int chunk_size = parameters.chunk_size;

    // four reconstruction parameters: T1, T2, rho_x, rho_y
    auto JHv = Array<cfloat, 2> {{4, nvoxels}};
    auto E_H = Array<ComputeT, 3> {{2, nvoxels, ns}};
    auto dEdT2_H = Array<ComputeT, 3> {{2, nvoxels, ns}};
    dim3 block_dim = {64, 4};

    // Initialize to zero
    ctx.parallel_device(
        {nvoxels},
        {chunk_size},
        [](kmm::DeviceResource& ctx, auto v) {  //
            ctx.fill(v, cfloat(0));
        },
        write(JHv));

    ctx.parallel_submit(
        {ns, nvoxels},
        {ns, chunk_size},
        kmm::GPUKernel(kernels::compute_sample_decay_hermitian<ComputeT>, block_dim),
        _xy,
        write(E_H),
        write(dEdT2_H),
        trajectory,
        parameters);

    for (int icoil = 0; icoil < ncoils; icoil++) {
        auto Ev = Array<float, 3> {{2, nreadouts, nvoxels}};
        auto dEdT2v = Array<float, 3> {{2, nreadouts, nvoxels}};
        auto vector_lo = Array<ComputeT, 3> {{2, nreadouts, ns}};

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto output, auto input) {
                convert_complex_to_planar(device, output, input.drop_axis(icoil));
            },
            write(vector_lo),
            vector);

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto result, auto lhs, auto rhs) {
                compute_complex_gemm(device, result, lhs, rhs, 1.0f, 0.0f, gemm);
            },
            write(Ev),
            vector_lo,
            E_H);

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto result, auto lhs, auto rhs) {
                compute_complex_gemm(device, result, lhs, rhs, 1.0f, 0.0f, gemm);
            },
            write(dEdT2v),
            vector_lo,
            dEdT2_H);

        ctx.parallel_submit(
            {nvoxels},
            {chunk_size},
            kmm::GPUKernel(kernels::jacobian_hermitian_product_finalize, 256),
            _x,
            nreadouts,
            icoil,
            write(JHv),
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            coil_sensitivities,
            Ev,
            dEdT2v);
    }

    return JHv;
}

Array<cfloat, 2> compute_jacobian_hermitian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 3> vector,
    JacobianComputeMethod kind) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int ncoils = int(coil_sensitivities.size(0));
    int nvoxels = parameters.nvoxels;

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.size(1) == nvoxels);
    COMPAS_CHECK(delta_echos_T1.size(0) == nreadouts);
    COMPAS_CHECK(delta_echos_T1.size(1) == nvoxels);
    COMPAS_CHECK(delta_echos_T2.size(0) == nreadouts);
    COMPAS_CHECK(delta_echos_T2.size(1) == nvoxels);
    COMPAS_CHECK(coil_sensitivities.size(0) == ncoils);
    COMPAS_CHECK(coil_sensitivities.size(1) == nvoxels);
    COMPAS_CHECK(vector.size(0) == ncoils);
    COMPAS_CHECK(vector.size(1) == nreadouts);
    COMPAS_CHECK(vector.size(2) == ns);

    if (kind == JacobianComputeMethod::Naive) {
        return compute_jacobian_hermitian_naive(
            ctx,
            nreadouts,
            ns,
            nvoxels,
            ncoils,
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector);
    } else if (kind == JacobianComputeMethod::Direct) {
        return compute_jacobian_hermitian_direct(
            ctx,
            nreadouts,
            ns,
            nvoxels,
            ncoils,
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector);
    } else if (kind == JacobianComputeMethod::GemmLow) {
        return compute_jacobian_hermitian_gemm<kernel_float::bfloat16_t>(
            ctx,
            GemmComputeMethod::Fast,
            nreadouts,
            ns,
            nvoxels,
            ncoils,
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector);
    } else {
        auto gemm = kind == JacobianComputeMethod::GemmFast ? GemmComputeMethod::Fast
                                                            : GemmComputeMethod::Regular;

        return compute_jacobian_hermitian_gemm(
            ctx,
            gemm,
            nreadouts,
            ns,
            nvoxels,
            ncoils,
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector);
    }
}

}  // namespace compas