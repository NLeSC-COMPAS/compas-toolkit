#include "constants.h"
#include "core/context.h"
#include "parameters/tissue.h"
#include "sequences/pssfp.h"
#include "simulate/sequence.h"
#include "simulate/signal.h"
#include "trajectories/cartesian.h"
#include "trajectories/multi.h"
#include "trajectories/spiral.h"

template<typename F>
auto catch_exceptions(F fun) -> decltype(fun()) {
    try {
        return fun();
    } catch (const std::exception& msg) {
        // Not sure how to pass the error to julia. Abort for now.
        fprintf(stderr, "COMPAS: fatal error occurred: %s\n", msg.what());
        std::abort();
    }
}

template<typename T, typename... Ns>
compas::host_view_mut<T, sizeof...(Ns)> make_view(T* ptr, Ns... sizes) {
    return {ptr, {{sizes...}}};
}

extern "C" const char* compas_version() {
    return COMPAS_VERSION;
}

extern "C" void compas_destroy(const compas::Object* obj) {
    return catch_exceptions([&] { delete obj; });
}

extern "C" const compas::CudaContext* compas_make_context(int device) {
    return catch_exceptions([&] {
        auto ctx = compas::make_context(device);
        return new compas::CudaContext(ctx);
    });
}

extern "C" void compas_destroy_context(const compas::CudaContext* ctx) {
    return catch_exceptions([&] { delete ctx; });
}

extern "C" compas::Trajectory* compas_make_cartesian_trajectory(
    const compas::CudaContext* context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::cfloat* k_start,
    compas::cfloat delta_k) {
    return catch_exceptions([&] {
        auto trajectory = compas::make_cartesian_trajectory(
            *context,
            nreadouts,
            samples_per_readout,
            delta_t,
            make_view(k_start, nreadouts),
            delta_k);

        return new compas::CartesianTrajectory(trajectory);
    });
}

extern "C" compas::Trajectory* compas_make_spiral_trajectory(
    const compas::CudaContext* context,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::cfloat* k_start,
    const compas::cfloat* delta_k) {
    return catch_exceptions([&] {
        auto trajectory = compas::make_spiral_trajectory(
            *context,
            nreadouts,
            samples_per_readout,
            delta_t,
            make_view(k_start, nreadouts),
            make_view(delta_k, nreadouts));

        return new compas::SpiralTrajectory(trajectory);
    });
}

extern "C" const compas::TissueParameters* compas_make_tissue_parameters(
    const compas::CudaContext* context,
    int nvoxels,
    const float* T1,
    const float* T2,
    const float* B1,
    const float* B0,
    const float* rho_x,
    const float* rho_y,
    const float* x,
    const float* y,
    const float* z) {
    return catch_exceptions([&] {
        auto params = compas::make_tissue_parameters(
            *context,
            nvoxels,
            make_view(T1, nvoxels),
            make_view(T2, nvoxels),
            make_view(B1, nvoxels),
            make_view(B0, nvoxels),
            make_view(rho_x, nvoxels),
            make_view(rho_y, nvoxels),
            make_view(x, nvoxels),
            make_view(y, nvoxels),
            make_view(z, nvoxels));

        return new compas::TissueParameters(params);
    });
}

extern "C" const compas::pSSFPSequence* compas_make_pssfp_sequence(
    const compas::CudaContext* context,
    int nRF,
    int nreadouts,
    int nslices,
    const compas::cfloat* RF_train,
    float TR,
    const compas::cfloat* gamma_dt_RF,
    float dt_ex,
    float dt_inv,
    float dt_pr,
    float gamma_dt_GRz_ex,
    float gamma_dt_GRz_inv,
    float gamma_dt_GRz_pr,
    const float* z) {
    return catch_exceptions([&] {
        auto seq = compas::make_pssfp_sequence(
            *context,
            make_view(RF_train, nreadouts),
            TR,
            make_view(gamma_dt_RF, nRF),
            {dt_ex, dt_inv, dt_pr},
            {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
            make_view(z, nslices));

        return new compas::pSSFPSequence(seq);
    });
}

extern "C" const compas::FISPSequence* compas_make_fisp_sequence(
    const compas::CudaContext* context,
    int nreadouts,
    int nslices,
    const compas::cfloat* RF_train,
    const compas::cfloat* slice_profiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    return catch_exceptions([&] {
        auto seq = compas::make_fisp_sequence(
            *context,
            make_view(RF_train, nreadouts),
            make_view(slice_profiles, nslices, nreadouts),
            TR,
            TE,
            max_state,
            TI);

        return new compas::FISPSequence(seq);
    });
}

extern "C" void compas_simulate_fisp_sequence(
    const compas::CudaContext* context,
    compas::cfloat* echos_ptr,
    const compas::TissueParameters* parameters,
    const compas::FISPSequence* sequence) {
    return catch_exceptions([&] {
        int nreadouts = sequence->RF_train.size();
        int nvoxels = parameters->nvoxels;

        auto echos = make_view(echos_ptr, nreadouts, nvoxels);
        auto d_echos = context->allocate<compas::cfloat>(echos.shape());

        compas::simulate_sequence(
            *context,
            d_echos.view_mut(),
            parameters->view(),
            sequence->view());

        d_echos.copy_to(echos);
    });
}

extern "C" void compas_simulate_pssfp_sequence(
    const compas::CudaContext* context,
    compas::cfloat* echos_ptr,
    const compas::TissueParameters* parameters,
    const compas::pSSFPSequence* sequence) {
    return catch_exceptions([&] {
        int nreadouts = sequence->RF_train.size();
        int nvoxels = parameters->nvoxels;

        auto echos = make_view(echos_ptr, nreadouts, nvoxels);
        auto d_echos = context->allocate<compas::cfloat>(echos.shape());

        compas::simulate_sequence(
            *context,
            d_echos.view_mut(),
            parameters->view(),
            sequence->view());

        d_echos.copy_to(echos);
    });
}

extern "C" void compas_simulate_signal(
    const compas::CudaContext* context,
    int ncoils,
    compas::cfloat* signal_ptr,
    const compas::cfloat* echos_ptr,
    compas::TissueParameters* parameters,
    compas::Trajectory* trajectory,
    const float* coils_ptr) {
    return catch_exceptions([&] {
        int nreadouts = trajectory->nreadouts;
        int samples_per_readout = trajectory->samples_per_readout;
        int nvoxels = parameters->nvoxels;

        auto signal = make_view(signal_ptr, ncoils, nreadouts, samples_per_readout);
        auto echos = make_view(echos_ptr, nreadouts, nvoxels);
        auto coils = make_view(coils_ptr, ncoils, nvoxels);

        auto d_signal = context->allocate(signal);
        auto d_echos = context->allocate(echos);
        auto d_coils = context->allocate(coils);

        compas::simulate_signal(
            *context,
            d_signal.view_mut(),
            d_echos.view(),
            parameters->view(),
            *trajectory,
            d_coils.view());

        d_signal.copy_to(signal);
    });
}