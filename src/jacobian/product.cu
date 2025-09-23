#include "compas/core/assertion.h"
#include "compas/core/utils.h"
#include "compas/jacobian/product.h"
#include "compas/utils/gemm.h"
#include "product_kernels.cuh"

namespace compas {

static Array<cfloat, 3> compute_jacobian_naive(
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
    Array<cfloat, 2> vector) {
    auto Jv = Array<cfloat, 3> {{ncoils, nreadouts, ns}};

    for (int icoil = 0; icoil < ncoils; icoil++) {
        ctx.parallel_kernel(
            {ns, nreadouts},
            {ns, nreadouts},
            {32, 8},
            kernels::jacobian_product_naive,
            nvoxels,
            icoil,
            nreadouts,
            ns,
            write(Jv),
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector);
    }

    return Jv;
}

template<
    int threads_per_item,
    int samples_per_thread,
    int readouts_per_thread,
    int coils_per_thread,
    int threads_per_block,
    int blocks_per_sm,
    typename... Args>
static void
launch_jacobian_product_impl(kmm::DeviceResource& ctx, kmm::Bounds<3> range, Args... args) {
    auto nsamples = range.size().y;
    auto nreadouts = range.size().z;

    if (nsamples % samples_per_thread != 0) {
        auto remainder = range;
        remainder.y.begin = range.y.end - nsamples % samples_per_thread;

        launch_jacobian_product_impl<
            threads_per_item,
            1,
            readouts_per_thread,
            coils_per_thread,
            threads_per_block,
            blocks_per_sm>(ctx, remainder, args...);

        nsamples -= nsamples % samples_per_thread;
        range.y.end = range.y.begin + nsamples;
    }

    if (nreadouts % readouts_per_thread != 0) {
        auto remainder = range;
        remainder.z.begin = range.z.end - nreadouts % readouts_per_thread;

        launch_jacobian_product_impl<
            threads_per_item,
            samples_per_thread,
            1,
            coils_per_thread,
            threads_per_block,
            blocks_per_sm>(ctx, remainder, args...);

        nreadouts -= nreadouts % readouts_per_thread;
        range.z.end = range.z.begin + nreadouts;
    }

    dim3 block_size = {threads_per_item, threads_per_block / (threads_per_item * 4), 4};
    dim3 grid_size = {
        1,
        div_ceil(uint(nsamples), block_size.y * samples_per_thread),
        div_ceil(uint(nreadouts), block_size.z * readouts_per_thread)};

    kernels::jacobian_product<
        threads_per_item,
        samples_per_thread,
        readouts_per_thread,
        coils_per_thread,
        threads_per_block,
        blocks_per_sm><<<grid_size, block_size, 0, ctx>>>(range, args...);
}

static void launch_jacobian_product(
    kmm::DeviceResource& ctx,
    kmm::Bounds<3> range,
    GPUSubviewMut<cfloat, 3> Jv,
    GPUSubview<cfloat, 2> coil_sensitivities,
    GPUSubview<cfloat, 2> E,
    GPUSubview<cfloat, 2> dEdT2,
    GPUSubview<cfloat, 2> adj_phase,
    GPUSubview<cfloat, 2> adj_decay) {
    auto coil_offset = 0;
    auto ncoils = coil_sensitivities.size(0);

#define COMPAS_COMPUTE_JACOBIAN_IMPL(N)                     \
    for (; coil_offset + N <= ncoils; coil_offset += N) {   \
        launch_jacobian_product_impl<16, 4, 4, N, 256, 16>( \
            ctx,                                            \
            range,                                          \
            coil_offset,                                    \
            Jv,                                             \
            coil_sensitivities,                             \
            E,                                              \
            dEdT2,                                          \
            adj_phase,                                      \
            adj_decay);                                     \
    }

    COMPAS_COMPUTE_JACOBIAN_IMPL(4)
    COMPAS_COMPUTE_JACOBIAN_IMPL(2)
    COMPAS_COMPUTE_JACOBIAN_IMPL(1)
}

static Array<cfloat, 3> compute_jacobian_direct(
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
    Array<cfloat, 2> vector) {
    using namespace kmm::placeholders;
    auto Jv = Array<cfloat, 3> {{ncoils, nreadouts, ns}};
    auto E = Array<cfloat, 2> {{ns, nvoxels}};
    auto dEdT2 = Array<cfloat, 2> {{ns, nvoxels}};
    auto adj_phase = Array<cfloat, 2> {{nreadouts, nvoxels}};
    auto adj_decay = Array<cfloat, 2> {{nreadouts, nvoxels}};

    auto chunk_size = parameters.chunk_size;
    auto _voxel = kmm::Axis(0);

    ctx.parallel_kernel(
        {nvoxels, ns},
        {chunk_size, ns},
        {32, 8},
        kernels::compute_sample_decay,
        _xy,
        write(E[_][_voxel]),
        write(dEdT2[_][_voxel]),
        trajectory,
        read(parameters, _voxel));

    ctx.parallel_kernel(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        {32, 8},
        kernels::compute_adjoint_sources,
        _xy,
        write(adj_phase[_][_voxel]),
        write(adj_decay[_][_voxel]),
        echos[_][_voxel],
        delta_echos_T1[_][_voxel],
        delta_echos_T2[_][_voxel],
        vector[_][_voxel],
        read(parameters, _voxel));

    ctx.parallel_submit(
        {nvoxels, ns, nreadouts},
        {chunk_size, ns, nreadouts},
        kmm::GPU(launch_jacobian_product),
        _xyz,
        write(Jv),
        coil_sensitivities[_][_voxel],
        E[_][_voxel],
        dEdT2[_][_voxel],
        adj_phase[_][_voxel],
        adj_decay[_][_voxel]);

    return Jv;
}

template<typename ComputeT = float>
static Array<cfloat, 3> compute_jacobian_gemm(
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
    Array<cfloat, 2> vector,
    GemmComputeMethod kind) {
    using namespace kmm::placeholders;
    auto Jv = Array<cfloat, 3> {{ncoils, nreadouts, ns}};
    auto E = Array<ComputeT, 3> {{2, ns, nvoxels}};
    auto dEdT2 = Array<ComputeT, 3> {{2, ns, nvoxels}};

    auto chunk_size = parameters.chunk_size;
    auto _voxel = kmm::Axis(0);

    ctx.parallel_kernel(
        {nvoxels, ns},
        {chunk_size, ns},
        {32, 8},
        kernels::compute_sample_decay_planar<ComputeT>,
        _xy,
        write(E[_][_][_voxel]),
        write(dEdT2[_][_][_voxel]),
        trajectory,
        read(parameters, _voxel));

    for (int icoil = 0; icoil < ncoils; icoil++) {
        auto Jv_coil = Array<float, 3> {{2, nreadouts, ns}};
        auto adj_phase = Array<ComputeT, 3> {{2, nreadouts, nvoxels}};
        auto adj_decay = Array<ComputeT, 3> {{2, nreadouts, nvoxels}};

        ctx.parallel_kernel(
            {nvoxels, nreadouts},
            {chunk_size, nreadouts},
            {32, 8},
            kernels::compute_adjoint_sources_with_coil<ComputeT>,
            _xy,
            write(adj_phase[_][_][_voxel]),
            write(adj_decay[_][_][_voxel]),
            icoil,
            coil_sensitivities[icoil][_voxel],
            echos[_][_voxel],
            delta_echos_T1[_][_voxel],
            delta_echos_T2[_][_voxel],
            vector[_][_voxel],
            read(parameters, _voxel));

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto result, auto lhs, auto rhs) {
                compute_complex_gemm(device, result, lhs, rhs, 1.0F, 0.0F, kind);
            },
            write(Jv_coil),
            adj_phase[_][_][_voxel],
            E[_][_][_voxel]);

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto result, auto lhs, auto rhs) {
                compute_complex_gemm(device, result, lhs, rhs, 1.0F, 1.0F, kind);
            },
            write(Jv_coil),
            adj_decay[_][_][_voxel],
            dEdT2[_][_][_voxel]);

        ctx.parallel_device(
            nvoxels,
            chunk_size,
            [=](auto& device, auto output, auto input) {
                convert_planar_to_complex(device, output.drop_axis(icoil), input);
            },
            write(Jv),
            Jv_coil);
    }

    return Jv;
}

Array<cfloat, 3> compute_jacobian(
    const CompasContext& ctx,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 2> vector,
    JacobianComputeMethod kind) {
    int ns = trajectory.samples_per_readout;
    int nreadouts = trajectory.nreadouts;
    int nvoxels = parameters.nvoxels;
    int ncoils = int(coil_sensitivities.size(0));

    COMPAS_CHECK(echos.size(0) == nreadouts);
    COMPAS_CHECK(echos.size(1) == nvoxels);
    COMPAS_CHECK(delta_echos_T1.size(0) == nreadouts);
    COMPAS_CHECK(delta_echos_T1.size(1) == nvoxels);
    COMPAS_CHECK(delta_echos_T2.size(0) == nreadouts);
    COMPAS_CHECK(delta_echos_T2.size(1) == nvoxels);
    COMPAS_CHECK(coil_sensitivities.size(0) == ncoils);
    COMPAS_CHECK(coil_sensitivities.size(1) == nvoxels);
    COMPAS_CHECK(vector.size(0) == 4);  // four reconstruction parameters: T1, T2, rho_x, rho_y
    COMPAS_CHECK(vector.size(1) == nvoxels);

    if (kind == JacobianComputeMethod::Naive) {
        return compute_jacobian_naive(
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
        return compute_jacobian_direct(
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
        return compute_jacobian_gemm<kernel_float::bfloat16_t>(
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
            vector,
            GemmComputeMethod::Fast);
    } else {
        auto gemm = kind == JacobianComputeMethod::GemmFast ? GemmComputeMethod::Fast
                                                            : GemmComputeMethod::Regular;

        return compute_jacobian_gemm(
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
            vector,
            gemm);
    }
}

}  // namespace compas