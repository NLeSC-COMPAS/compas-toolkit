#include "compas/jacobian/hermitian.h"
#include "compas/jacobian/product.h"
#include "compas/parameters/tissue.h"
#include "compas/simulate/derivative.h"
#include "compas/simulate/fisp.h"
#include "compas/simulate/pssfp.h"
#include "compas/trajectories/cartesian.h"
#include "compas/trajectories/phase_encoding.h"
#include "compas/trajectories/signal.h"
#include "compas/trajectories/spiral.h"
#include "compas/trajectories/trajectory.h"
#include "compas/utils/residual.h"
#include "constants.h"

// Alias for complex float.
using cfloat = compas::complex_type<float>;

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
    using index_type = typename compas::host_view_mut<T, sizeof...(Ns)>::index_type;
    return {ptr, {{kmm::checked_cast<index_type>(sizes)...}}};
}

extern "C" const char* compas_version() {
    return COMPAS_VERSION;
}

extern "C" void compas_destroy(const compas::Object* obj) {
    return catch_exceptions([&] { delete obj; });
}

extern "C" const compas::CompasContext* compas_make_context(int device) {
    return catch_exceptions([&] {
        auto ctx = compas::make_context(device);
        return new compas::CompasContext(ctx);
    });
}

extern "C" void compas_destroy_context(const compas::CompasContext* ctx) {
    return catch_exceptions([&] { delete ctx; });
}

extern "C" const kmm::ArrayBase* compas_make_array_float(
    const compas::CompasContext* context,
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
    const compas::CompasContext* context,
    const kmm::ArrayBase* input_array,
    float* dest_ptr,
    int64_t length) {
    catch_exceptions([&]() {
        size_t num_bytes = kmm::checked_mul(kmm::checked_cast<size_t>(length), sizeof(float));
        input_array->copy_bytes_to(dest_ptr, num_bytes);
    });
}

extern "C" const kmm::ArrayBase* compas_make_array_complex(
    const compas::CompasContext* context,
    const cfloat* data_ptr,
    int rank,
    int64_t* sizes) {
    return catch_exceptions([&]() -> kmm::ArrayBase* {
        if (rank == 1) {
            return new kmm::Array<cfloat>(context->allocate(data_ptr, sizes[0]));
        } else if (rank == 2) {
            return new kmm::Array<cfloat, 2>(context->allocate(data_ptr, sizes[0], sizes[1]));
        } else if (rank == 3) {
            return new kmm::Array<cfloat, 3>(
                context->allocate(data_ptr, sizes[0], sizes[1], sizes[2]));
        } else {
            COMPAS_PANIC("cannot support rank > 3");
        }
    });
}

extern "C" void compas_read_array_complex(
    const compas::CompasContext* context,
    const kmm::ArrayBase* input_array,
    cfloat* dest_ptr,
    int64_t length) {
    catch_exceptions([&]() {
        size_t num_bytes = kmm::checked_mul(kmm::checked_cast<size_t>(length), 2 * sizeof(float));
        input_array->copy_bytes_to(dest_ptr, num_bytes);
    });
}

extern "C" void compas_destroy_array(const kmm::ArrayBase* array) {
    return catch_exceptions([&] { delete array; });
}

extern "C" const compas::TissueParameters* compas_make_tissue_parameters(
    const compas::CompasContext* context,
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
    const compas::CompasContext* context,
    const compas::TissueParameters* parameters,
    compas::Array<cfloat>* RF_train,
    compas::Array<cfloat, 2>* sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    return catch_exceptions([&] {
        auto sequence = compas::FISPSequence {*RF_train, *sliceprofiles, TR, TE, max_state, TI};

        auto echos = compas::simulate_magnetization(*context, *parameters, sequence);

        return new kmm::Array<cfloat, 2> {echos};
    });
}

extern "C" kmm::ArrayBase* compas_simulate_magnetization_pssfp(
    const compas::CompasContext* context,
    const compas::TissueParameters* parameters,
    const compas::Array<cfloat>* RF_train,
    float TR,
    const compas::Array<cfloat>* gamma_dt_RF,
    float dt_ex,
    float dt_inv,
    float dt_pr,
    float gamma_dt_GRz_ex,
    float gamma_dt_GRz_inv,
    float gamma_dt_GRz_pr,
    const compas::Array<float>* z) {
    return catch_exceptions([&] {
        auto sequence = compas::pSSFPSequence {
            *RF_train,
            TR,
            *gamma_dt_RF,
            {dt_ex, dt_inv, dt_pr},
            {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
            *z};

        auto echos = compas::simulate_magnetization(*context, *parameters, sequence);

        return new kmm::Array<cfloat, 2> {echos};
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_cartesian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    const compas::Array<cfloat, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<cfloat>* k_start,
    cfloat delta_k) {
    return catch_exceptions([&] {
        auto trajectory = compas::CartesianTrajectory {
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k};

        auto signal =
            compas::magnetization_to_signal(*context, *echos, *parameters, trajectory, *coils);
        return new compas::Array<cfloat, 3>(signal);
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_spiral(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    const compas::Array<cfloat, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<cfloat>* k_start,
    const compas::Array<cfloat>* delta_k) {
    return catch_exceptions([&] {
        auto trajectory =
            compas::SpiralTrajectory {nreadouts, samples_per_readout, delta_t, *k_start, *delta_k};

        auto signal =
            compas::magnetization_to_signal(*context, *echos, *parameters, trajectory, *coils);
        return new compas::Array<cfloat, 3>(signal);
    });
}

extern "C" compas::Array<cfloat, 3>* compas_compute_jacobian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::Array<cfloat, 2>* delta_echos_T1,
    const compas::Array<cfloat, 2>* delta_echos_T2,
    const compas::TissueParameters* parameters,
    const compas::Array<cfloat, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<cfloat>* k_start,
    cfloat delta_k,
    const compas::Array<cfloat, 2>* vector) {
    return catch_exceptions([&] {
        auto trajectory = compas::CartesianTrajectory {
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k};

        auto Jv = compas::compute_jacobian(
            *context,
            *echos,
            *delta_echos_T1,
            *delta_echos_T2,
            *parameters,
            trajectory,
            *coils,
            *vector);

        return new compas::Array<cfloat, 3>(Jv);
    });
}

extern "C" compas::Array<cfloat, 2>* compas_compute_jacobian_hermitian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::Array<cfloat, 2>* delta_echos_T1,
    const compas::Array<cfloat, 2>* delta_echos_T2,
    const compas::TissueParameters* parameters,
    const compas::Array<cfloat, 2>* coils,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<cfloat>* k_start,
    cfloat delta_k,
    const compas::Array<cfloat, 3>* vector) {
    return catch_exceptions([&] {
        auto trajectory = compas::CartesianTrajectory {
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k};

        auto d_JHv = compas::compute_jacobian_hermitian(
            *context,
            *echos,
            *delta_echos_T1,
            *delta_echos_T2,
            *parameters,
            trajectory,
            *coils,
            *vector);

        return new compas::Array<cfloat, 2>(d_JHv);
    });
}

extern "C" compas::Array<cfloat, 2>* phase_encoding(
    const compas::CompasContext* context,
    const compas::Array<cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    const compas::Array<cfloat>* k_start,
    cfloat delta_k) {
    return catch_exceptions([&] {
        auto trajectory = compas::CartesianTrajectory {
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k};

        auto d_echos = compas::phase_encoding(*context, *echos, *parameters, trajectory);

        return new compas::Array<cfloat, 2>(d_echos);
    });
}

extern "C" compas::Array<cfloat, 3>* compas_compute_residual(
    const compas::CompasContext* context,
    const compas::Array<cfloat, 3>* lhs,
    const compas::Array<cfloat, 3>* rhs,
    float* sum) {
    return catch_exceptions([&] {
        auto diff = compas::compute_residual(*context, *lhs, *rhs, sum);

        return new compas::Array<cfloat, 3>(diff);
    });
}

extern "C" compas::Array<cfloat, 2>* compas_simulate_magnetization_derivative_pssfp(
    const compas::CompasContext* context,
    int field,
    const compas::Array<cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    float delta,
    const compas::Array<cfloat>* RF_train,
    float TR,
    const compas::Array<cfloat>* gamma_dt_RF,
    float dt_ex,
    float dt_inv,
    float dt_pr,
    float gamma_dt_GRz_ex,
    float gamma_dt_GRz_inv,
    float gamma_dt_GRz_pr,
    const compas::Array<float>* z) {
    return catch_exceptions([&] {
        auto sequence = compas::pSSFPSequence {
            *RF_train,
            TR,
            *gamma_dt_RF,
            {dt_ex, dt_inv, dt_pr},
            {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
            *z};

        auto delta_echos = compas::simulate_magnetization_derivative(
            *context,
            field,
            *echos,
            *parameters,
            sequence,
            delta);

        return new compas::Array<cfloat, 2> {delta_echos};
    });
}

extern "C" compas::Array<cfloat, 2>* compas_simulate_magnetization_derivative_fisp(
    const compas::CompasContext* context,
    int field,
    const compas::Array<cfloat, 2>* echos,
    const compas::TissueParameters* parameters,
    float delta,
    compas::Array<cfloat>* RF_train,
    compas::Array<cfloat, 2>* sliceprofiles,
    float TR,
    float TE,
    int max_state,
    float TI) {
    return catch_exceptions([&] {
        auto sequence = compas::FISPSequence {*RF_train, *sliceprofiles, TR, TE, max_state, TI};

        auto delta_echos = compas::simulate_magnetization_derivative(
            *context,
            field,
            *echos,
            *parameters,
            sequence,
            delta);

        return new compas::Array<cfloat, 2> {delta_echos};
    });
}