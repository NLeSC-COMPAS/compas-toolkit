#include "constants.h"
#include "core/context.h"
#include "jacobian/product.h"
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
    } catch (...) {
        fprintf(stderr, "COMPAS: fatal error occurred: %s\n", "unknown exception");
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
    float* dest_ptr,
    int64_t length) {
    catch_exceptions([&]() {
        size_t num_bytes = kmm::checked_mul(kmm::checked_cast<size_t>(length), sizeof(float));
        input_array->read_bytes(dest_ptr, num_bytes);
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
    compas::cfloat* dest_ptr,
    int64_t length) {
    catch_exceptions([&]() {
        size_t num_bytes = kmm::checked_mul(kmm::checked_cast<size_t>(length), 2 * sizeof(float));
        input_array->read_bytes(dest_ptr, num_bytes);
    });
}

extern "C" void compas_destroy_array(const kmm::ArrayBase* array) {
    return catch_exceptions([&] { delete array; });
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

extern "C" kmm::ArrayBase* compas_simulate_magnetization_fisp(
    const compas::CudaContext* context,
    const compas::TissueParameters* parameters,
    compas::Array<compas::cfloat>* RF_train,
    compas::Array<compas::cfloat, 2>* sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    return catch_exceptions([&] {
        int nreadouts = RF_train->size();
        int nvoxels = parameters->nvoxels;

        auto* echos = new compas::Array<compas::cfloat, 2>(nreadouts, nvoxels);
        auto sequence = compas::FISPSequence {*RF_train, *sliceprofiles, TR, TE, max_state, TI};

        context->submit_device(
            compas::simulate_magnetization_fisp,
            write(*echos),
            *parameters,
            sequence);

        return echos;
    });
}

extern "C" kmm::ArrayBase* compas_simulate_magnetization_pssfp(
    const compas::CudaContext* context,
    const compas::TissueParameters* parameters,
    const compas::Array<compas::cfloat>* RF_train,
    float TR,
    const compas::Array<compas::cfloat>* gamma_dt_RF,
    float dt_ex,
    float dt_inv,
    float dt_pr,
    float gamma_dt_GRz_ex,
    float gamma_dt_GRz_inv,
    float gamma_dt_GRz_pr,
    const compas::Array<float>* z) {
    return catch_exceptions([&] {
        int nreadouts = RF_train->size();
        int nvoxels = parameters->nvoxels;
        auto* echos = new compas::Array<compas::cfloat, 2>(nreadouts, nvoxels);

        auto sequence = compas::pSSFPSequence {
            *RF_train,
            TR,
            *gamma_dt_RF,
            {dt_ex, dt_inv, dt_pr},
            {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
            *z};

        context->submit_device(
            compas::simulate_magnetization_pssfp,
            write(*echos),
            *parameters,
            sequence);

        return echos;
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_cartesian(
    const compas::CudaContext* context,
    int ncoils,
    const compas::Array<compas::cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    const compas::Array<float, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<compas::cfloat>* k_start,
    compas::cfloat delta_k) {
    return catch_exceptions([&] {
        auto trajectory = compas::CartesianTrajectory {
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k};

        auto signal =
            compas::magnetization_to_signal(*context, *echos, *parameters, trajectory, *coils);
        return new compas::Array<compas::cfloat, 3>(signal);
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_spiral(
    const compas::CudaContext* context,
    int ncoils,
    const compas::Array<compas::cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    const compas::Array<float, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<compas::cfloat>* k_start,
    const compas::Array<compas::cfloat>* delta_k) {
    return catch_exceptions([&] {
        auto trajectory =
            compas::SpiralTrajectory {nreadouts, samples_per_readout, delta_t, *k_start, *delta_k};

        auto signal =
            compas::magnetization_to_signal(*context, *echos, *parameters, trajectory, *coils);
        return new compas::Array<compas::cfloat, 3>(signal);
    });
}

extern "C" void compas_compute_jacobian(
    const compas::CudaContext* context,
    int ncoils,
    compas::cfloat* Jv_ptr,
    const compas::cfloat* echos_ptr,
    const compas::cfloat* delta_echos_ptr,
    const compas::TissueParameters* parameters,
    const compas::CartesianTrajectory* trajectory,
    const float* coils_ptr,
    const compas::cfloat* vector_ptr) {
    return catch_exceptions([&] {
        int nreadouts = trajectory->nreadouts;
        int ns = trajectory->samples_per_readout;
        int nvoxels = parameters->nvoxels;

        auto Jv = make_view(Jv_ptr, ncoils, nreadouts * ns);

        auto d_echos = context->allocate(make_view(echos_ptr, nreadouts, nvoxels));
        auto d_delta_echos = context->allocate(make_view(delta_echos_ptr, 2, nreadouts, nvoxels));

        auto d_coils = context->allocate(make_view(coils_ptr, ncoils, nvoxels));
        auto d_vector = context->allocate(make_view(vector_ptr, 4, nvoxels));

        auto d_Jv = compas::compute_jacobian(
            *context,
            d_echos,
            d_delta_echos,
            *parameters,
            *trajectory,
            d_coils,
            d_vector);

        d_Jv.read(Jv);
    });
}

extern "C" void compas_compute_jacobian_hermitian(
    const compas::CudaContext* context,
    int ncoils,
    compas::cfloat* JHv_ptr,
    const compas::cfloat* echos_ptr,
    const compas::cfloat* delta_echos_ptr,
    const compas::TissueParameters* parameters,
    const compas::CartesianTrajectory* trajectory,
    const float* coils_ptr,
    const compas::cfloat* vector_ptr) {
    return catch_exceptions([&] {
        int nreadouts = trajectory->nreadouts;
        int ns = trajectory->samples_per_readout;
        int nvoxels = parameters->nvoxels;

        auto JHv = make_view(JHv_ptr, 4, nvoxels);

        auto d_echos = context->allocate(make_view(echos_ptr, nreadouts, nvoxels));
        auto d_delta_echos = context->allocate(make_view(delta_echos_ptr, 2, nreadouts, nvoxels));

        auto d_coils = context->allocate(make_view(coils_ptr, ncoils, nvoxels));
        auto d_vector = context->allocate(make_view(vector_ptr, ncoils, nreadouts * ns));

        auto d_JHv = compas::compute_jacobian_hermitian(
            *context,
            d_echos,
            d_delta_echos,
            *parameters,
            *trajectory,
            d_coils,
            d_vector);

        d_JHv.read(JHv);
    });
}