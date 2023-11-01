#include <jlcxx/jlcxx.hpp>

#include "core/context.h"
#include "parameters/tissue.h"
#include "sequences/pssfp.h"
#include "simulate/sequence.h"
#include "simulate/signal.h"
#include "trajectories/multi.h"

template<typename T, int N>
compas::vector<int, N> to_shape(const jlcxx::ArrayRef<T, N>& array) {
    compas::vector<int, N> shape;
    for (int i = 0; i < N; i++) {
        shape[i] = jl_array_dim(array.wrapped(), N - i - 1);
    }
    return shape;
}

template<typename T, int N>
compas::host_view<T, N> into_view(const jlcxx::ArrayRef<T, N>& array) {
    return {array.data(), to_shape(array)};
}

template<typename T, int N>
compas::host_view_mut<T, N> into_view_mut(jlcxx::ArrayRef<T, N>& array) {
    return {array.data(), to_shape(array)};
}

template<typename T, int N>
compas::host_view<compas::complex_type<T>, N>
into_view(const jlcxx::ArrayRef<std::complex<T>, N>& array) {
    auto ptr = static_cast<const compas::complex_type<T>*>(static_cast<const void*>(array.data()));
    return {ptr, to_shape(array)};
}

template<typename T, int N>
compas::host_view_mut<compas::complex_type<T>, N>
into_view_mut(jlcxx::ArrayRef<std::complex<T>, N>& array) {
    auto ptr = static_cast<compas::complex_type<T>*>(static_cast<void*>(array.data()));
    return {ptr, to_shape(array)};
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<compas::CudaContext>("CudaContext");
    mod.method("make_context", compas::make_context);

    mod.add_type<compas::Trajectory>("Trajectory");
    mod.method(
        "make_spiral_trajectory",
        [](const compas::CudaContext& context,
           int nreadouts,
           int samples_per_readout,
           float delta_t,
           jlcxx::ArrayRef<std::complex<float>> k_start,
           jlcxx::ArrayRef<std::complex<float>> delta_k) -> compas::Trajectory {
            return make_spiral_trajectory(
                context,
                nreadouts,
                samples_per_readout,
                delta_t,
                into_view(k_start),
                into_view(delta_k));
        });

    mod.method(
        "make_cartesian_trajectory",
        [](const compas::CudaContext& context,
           int nreadouts,
           int samples_per_readout,
           float delta_t,
           jlcxx::ArrayRef<std::complex<float>> k_start,
           std::complex<float> delta_k) -> compas::Trajectory {
            return make_cartesian_trajectory(
                context,
                nreadouts,
                samples_per_readout,
                delta_t,
                into_view(k_start),
                {delta_k.real(), delta_k.imag()});
        });

    mod.add_type<compas::TissueParameters>("TissueParameters");
    mod.method(
        "make_tissue_parameters",
        [](const compas::CudaContext& context,
           int nvoxels,
           jlcxx::ArrayRef<float> T1,
           jlcxx::ArrayRef<float> T2,
           jlcxx::ArrayRef<float> B1,
           jlcxx::ArrayRef<float> B0,
           jlcxx::ArrayRef<float> rho_x,
           jlcxx::ArrayRef<float> rho_y,
           jlcxx::ArrayRef<float> x,
           jlcxx::ArrayRef<float> y,
           jlcxx::ArrayRef<float> z) {
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
                into_view(z));
        });

    mod.method(
        "make_tissue_parameters",
        [](const compas::CudaContext& context,
           int nvoxels,
           jlcxx::ArrayRef<float> T1,
           jlcxx::ArrayRef<float> T2,
           jlcxx::ArrayRef<float> B1,
           jlcxx::ArrayRef<float> B0,
           jlcxx::ArrayRef<float> rho_x,
           jlcxx::ArrayRef<float> rho_y,
           jlcxx::ArrayRef<float> x,
           jlcxx::ArrayRef<float> y) {
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
                {});
        });

    mod.method(
        "simulate_signal",
        [](const compas::CudaContext& context,
           jlcxx::ArrayRef<std::complex<float>, 3> julia_signal,
           jlcxx::ArrayRef<std::complex<float>, 2> julia_echos,
           compas::TissueParameters parameters,
           compas::Trajectory trajectory,
           jlcxx::ArrayRef<float, 2> julia_coil_sensitivities) {
            auto signal = into_view_mut(julia_signal);
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

            d_signal.copy_to(signal);
        });

    mod.add_type<compas::pSSFPSequence>("pSSFPSequence");
    mod.method(
        "make_pssfp_sequence",
        [](const compas::CudaContext& context,
           jlcxx::ArrayRef<std::complex<float>> RF_train,
           float TR,
           jlcxx::ArrayRef<std::complex<float>> gamma_dt_RF,
           jlcxx::ArrayRef<float> dt,
           jlcxx::ArrayRef<float> gamma_dt_GRz,
           jlcxx::ArrayRef<float> z) {
            COMPAS_ASSERT(dt.size() == 3);
            COMPAS_ASSERT(gamma_dt_GRz.size() == 3);

            return make_pssfp_sequence(
                context,
                into_view(RF_train),
                TR,
                into_view(gamma_dt_RF),
                {dt[0], dt[1], dt[2]},
                {gamma_dt_GRz[0], gamma_dt_GRz[1], gamma_dt_GRz[2]},
                into_view(z));
        });

    mod.method(
        "simulate_sequence",
        [](const compas::CudaContext& context,
           jlcxx::ArrayRef<std::complex<float>, 2> julia_echos,
           compas::TissueParameters parameters,
           compas::pSSFPSequence sequence) {
            auto echos = into_view_mut(julia_echos);
            auto d_echos = context.allocate<compas::cfloat>(echos.shape());

            compas::simulate_sequence(context, d_echos, parameters, sequence);

            d_echos.copy_to(echos);
        });

    mod.add_type<compas::FISPSequence>("FISPSequence");
    mod.method(
        "make_fisp_sequence",
        [](const compas::CudaContext& context,
           jlcxx::ArrayRef<std::complex<float>> RF_train,
           jlcxx::ArrayRef<std::complex<float>, 2> slice_profiles,
           float TR,
           float TE,
           int max_state,
           float TI) {
            return make_fisp_sequence(
                context,
                into_view(RF_train),
                into_view(slice_profiles),
                TR,
                TE,
                max_state,
                TI);
        });

    mod.method(
            "simulate_sequence",
            [](const compas::CudaContext& context,
               jlcxx::ArrayRef<std::complex<float>, 2> julia_echos,
               compas::TissueParameters parameters,
               compas::FISPSequence sequence) {
                auto echos = into_view_mut(julia_echos);
                auto d_echos = context.allocate<compas::cfloat>(echos.shape());

                compas::simulate_sequence(context, d_echos, parameters, sequence);

                d_echos.copy_to(echos);
            });

    mod.method(
            "simulate_sequence_raw",
            [](const compas::CudaContext& context,
               long d_echos_ptr,
               compas::TissueParameters parameters,
               compas::FISPSequence sequence) {
                int nvoxels = parameters.nvoxels;
                int nreadouts = sequence.RF_train.size();

                auto d_echos = context.from_raw_pointer<compas::cfloat, 2>(d_echos_ptr, {
                    nreadouts,
                    nvoxels
                });
                compas::simulate_sequence(context, d_echos, parameters, sequence);
            });

    mod.add_type<compas::FISP3DSequence>("FISP3DSequence");
    mod.method(
        "make_fisp3d_sequence",
        [](const compas::CudaContext& context,
           jlcxx::ArrayRef<std::complex<float>> RF_train,
           float TR,
           float TE,
           int max_state,
           float TI,
           float TW) {
            return make_fisp3d_sequence(
                context,
                into_view(RF_train),
                TR,
                TE,
                max_state,
                TI,
                TW);
        });

    mod.method(
            "simulate_sequence",
            [](const compas::CudaContext& context,
               jlcxx::ArrayRef<std::complex<float>, 2> julia_echos,
               compas::TissueParameters parameters,
               compas::FISP3DSequence sequence) {
                auto echos = into_view_mut(julia_echos);
                auto d_echos = context.allocate<compas::cfloat>(echos.shape());

                compas::simulate_sequence(context, d_echos, parameters, sequence);

                d_echos.copy_to(echos);
            });

    mod.method(
            "simulate_sequence_raw",
            [](const compas::CudaContext& context,
               long d_echos_ptr,
               compas::TissueParameters parameters,
               compas::FISP3DSequence sequence) {
                int nvoxels = parameters.nvoxels;
                int nreadouts = sequence.RF_train.size();

                auto d_echos = context.from_raw_pointer<compas::cfloat, 2>(d_echos_ptr, {
                    nreadouts,
                    nvoxels
                });
                compas::simulate_sequence(context, d_echos, parameters, sequence);
            });
}