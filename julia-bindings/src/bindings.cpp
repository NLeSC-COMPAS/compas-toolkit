#include "context.h"
#include "trajectory.h"
#include "jlcxx/jlcxx.hpp"
#include "tissueparameters.h"
#include "simulate_signal.h"

template <typename T, int N>
compas::host_view<T, N> into_view(const jlcxx::ArrayRef<T, N>& array) {
    compas::fixed_array<int, N> shape;
    for (int i = 0; i < N; i++) {
        shape[i] = jl_array_dim(array.wrapped(), i);
    }

    return {array.data(), shape};
}

template <typename T, int N>
compas::host_view<compas::complex_type<T>, N> into_view(const jlcxx::ArrayRef<std::complex<T>, N>& array) {
    compas::fixed_array<int, N> shape;

    for (int i = 0; i < N; i++) {
        shape[i] = jl_array_dim(array.wrapped(), N - i - 1);
    }

    auto ptr = static_cast<const compas::complex_type<T>*>(static_cast<const void*>(array.data()));
    return {ptr, shape};
}


JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<compas::CudaContext>("CudaContext");
    mod.method("make_context", compas::make_context);


    mod.add_type<compas::CartesianTrajectory>("CartesianTrajectory");
    mod.method("make_cartesian_trajectory", [](
        const compas::CudaContext& context,
        int nreadouts,
        int samples_per_readout,
        float delta_t,
        jlcxx::ArrayRef<std::complex<float>> k_start,
        jlcxx::ArrayRef<std::complex<float>> delta_k
    ) {
        return make_cartesian_trajectory(
                context, nreadouts, samples_per_readout,
                delta_t,
                into_view(k_start),
                into_view(delta_k)
        );
    });

    mod.add_type<compas::TissueParameters>("TissueParameters");
    mod.method("make_tissue_parameters", [](
            const compas::CudaContext& context,
            int nvoxels,
            jlcxx::ArrayRef<float> T1,
            jlcxx::ArrayRef<float> T2,
            jlcxx::ArrayRef<float> B1,
            jlcxx::ArrayRef<float> B0,
            jlcxx::ArrayRef<float> rho_x,
            jlcxx::ArrayRef<float> rho_y,
            jlcxx::ArrayRef<float> x,
            jlcxx::ArrayRef<float> y,
            jlcxx::ArrayRef<float> z
    ) {
        return make_tissue_parameters(
                context,
                nvoxels,
                into_view(T1),
                into_view(T2),
                into_view(B1),
                into_view(B0),
                into_view(rho_x),
                into_view(rho_y),
                into_view(x),
                into_view(y),
                into_view(z)
            );
    });

    mod.method("make_tissue_parameters", [](
            const compas::CudaContext& context,
            int nvoxels,
            jlcxx::ArrayRef<float> T1,
            jlcxx::ArrayRef<float> T2,
            jlcxx::ArrayRef<float> B1,
            jlcxx::ArrayRef<float> B0,
            jlcxx::ArrayRef<float> rho_x,
            jlcxx::ArrayRef<float> rho_y,
            jlcxx::ArrayRef<float> x,
            jlcxx::ArrayRef<float> y
    ) {
        return make_tissue_parameters(
                context,
                nvoxels,
                into_view(T1),
                into_view(T2),
                into_view(B1),
                into_view(B0),
                into_view(rho_x),
                into_view(rho_y),
                into_view(x),
                into_view(y),
                {}
        );
    });


    mod.method("simulate_signal", [](
            const compas::CudaContext& context,
            jlcxx::ArrayRef<std::complex<float>, 3> julia_signal,
            jlcxx::ArrayRef<std::complex<float>, 2> julia_echos,
            compas::TissueParameters parameters,
            compas::CartesianTrajectory trajectory,
            jlcxx::ArrayRef<std::complex<float>, 2> julia_coil_sensitivities
    ) {
        auto signal = into_view(julia_signal);
        auto echos = into_view(julia_echos);
        auto coil_sensitivities = into_view(julia_coil_sensitivities);

        auto d_signal = context.allocate(signal);
        auto d_echos = context.allocate(echos);
        auto d_coil_sensitivities = context.allocate(coil_sensitivities);

        compas::simulate_signal(
                context,
                d_signal,
                d_echos,
                parameters,
                trajectory,
                d_coil_sensitivities);

//        d_signal.copy_to(signal);
    });
}