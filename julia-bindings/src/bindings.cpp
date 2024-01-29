#include "constants.h"
#include "core/context.h"
#include "parameters/tissue.h"
#include "sequences/pssfp.h"
#include "simulate/sequence.h"
#include "simulate/signal.h"
#include "trajectories/cartesian.h"
#include "trajectories/spiral.h"
#include "trajectories/trajectory.h"

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
    return {ptr, kmm::fixed_array<kmm::index_t, sizeof...(Ns)> {sizes...}};
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

extern "C" const kmm::ArrayBase* compas_make_array_float(
    const compas::CudaContext* context,
    const float* data_ptr,
    int rank,
    int64_t* sizes) {
    return catch_exceptions([&]() -> kmm::ArrayBase* {
        if (rank == 1) {
            return new kmm::Array<float>(context->allocate(data_ptr, sizes[0]));
        } else if (rank == 2) {
            return new kmm::Array<float, 2>(context->allocate(data_ptr, sizes[0], sizes[1]));
        } else if (rank == 3) {
            return new kmm::Array<float, 3>(
                context->allocate(data_ptr, sizes[0], sizes[1], sizes[2]));
        } else {
            COMPAS_PANIC("cannot support rank > 3");
        }
    });
}

extern "C" void compas_read_array_float(
    const compas::CudaContext* context,
    const kmm::ArrayBase* input_array,
    float* dest_ptr) {
    catch_exceptions([&]() -> kmm::ArrayBase* {
        input_array->block()->read(dest_ptr, input_array->size() * sizeof(float));
    });
}

extern "C" const kmm::ArrayBase* compas_make_array_complex(
    const compas::CudaContext* context,
    const compas::cfloat* data_ptr,
    int rank,
    int64_t* sizes) {
    return catch_exceptions([&]() -> kmm::ArrayBase* {
        if (rank == 1) {
            return new kmm::Array<compas::cfloat>(context->allocate(data_ptr, sizes[0]));
        } else if (rank == 2) {
            return new kmm::Array<compas::cfloat, 2>(
                context->allocate(data_ptr, sizes[0], sizes[1]));
        } else if (rank == 3) {
            return new kmm::Array<compas::cfloat, 3>(
                context->allocate(data_ptr, sizes[0], sizes[1], sizes[2]));
        } else {
            COMPAS_PANIC("cannot support rank > 3");
        }
    });
}

extern "C" void compas_read_array_complex(
    const compas::CudaContext* context,
    const kmm::ArrayBase* input_array,
    compas::cfloat* dest_ptr) {
    catch_exceptions([&]() -> kmm::ArrayBase* {
        input_array->block()->read(dest_ptr, input_array->size() * sizeof(compas::cfloat));
    });
}

extern "C" void compas_destroy_array(const kmm::ArrayBase* array) {
    return catch_exceptions([&] { delete array; });
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

extern "C" void compas_simulate_magnetization_fisp(
    const compas::CudaContext* context,
    compas::cfloat* echos_ptr,
    const compas::TissueParameters* parameters,
    compas::CudaArray<compas::cfloat>* RF_train,
    compas::CudaArray<compas::cfloat, 2>* sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI
    ) {
    return catch_exceptions([&] {
        int nreadouts = RF_train->size();
        int nvoxels = parameters->nvoxels;

        auto echos = make_view(echos_ptr, nreadouts, nvoxels);
        auto d_echos = compas::CudaArray<compas::cfloat, 2>(nreadouts, nvoxels);

        auto sequence = compas::FISPSequence {
                *RF_train,
                *sliceprofiles,
                TR,
                TE,
                max_state,
                TI
        };

        context->submit_device(
            compas::simulate_magnetization_fisp,
            write(d_echos),
            *parameters,
            sequence);

        d_echos.read(echos);
    });
}

extern "C" void compas_simulate_magnetization_pssfp(
    const compas::CudaContext* context,
    compas::cfloat* echos_ptr,
    const compas::TissueParameters* parameters,
    const compas::CudaArray<compas::cfloat>* RF_train,
    float TR,
    const compas::CudaArray<compas::cfloat>* gamma_dt_RF,
    float dt_ex,
    float dt_inv,
    float dt_pr,
    float gamma_dt_GRz_ex,
    float gamma_dt_GRz_inv,
    float gamma_dt_GRz_pr,
    const compas::CudaArray<float>* z) {
    return catch_exceptions([&] {
        int nreadouts = RF_train->size();
        int nvoxels = parameters->nvoxels;

        auto echos = make_view(echos_ptr, nreadouts, nvoxels);
        auto d_echos = context->allocate(echos);

        auto sequence = compas::pSSFPSequence {
                *RF_train,
                TR,
                *gamma_dt_RF,
                {dt_ex, dt_inv, dt_pr},
                {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
                *z
        };

        context->submit_device(
            compas::simulate_magnetization_pssfp,
            write(d_echos),
            *parameters,
            sequence);

        d_echos.read(echos);
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal(
    const compas::CudaContext* context,
    int ncoils,
    const compas::CudaArray<compas::cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    const compas::Trajectory* trajectory,
    const float* coils_ptr) {
    return catch_exceptions([&] {
        int nreadouts = trajectory->nreadouts;
        int samples_per_readout = trajectory->samples_per_readout;
        int nvoxels = parameters->nvoxels;

        auto coils = make_view(coils_ptr, ncoils, nvoxels);

        auto d_coils = context->allocate(coils);

        auto signal =
            compas::magnetization_to_signal(*context, *echos, *parameters, *trajectory, d_coils);

        return new compas::CudaArray<compas::cfloat, 3>(signal);
    });
}