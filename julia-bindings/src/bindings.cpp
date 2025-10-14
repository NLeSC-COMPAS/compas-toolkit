#include "common.h"
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

extern "C" const char* compas_version() {
    return COMPAS_VERSION;
}

extern "C" const compas::CompasContext* compas_make_context() {
    return catch_exceptions([&] {
        auto ctx = compas::make_context();
        return new compas::CompasContext(ctx);
    });
}

extern "C" const compas::CompasContext*
compas_copy_context_for_device(compas::CompasContext* ctx, int device) {
    return catch_exceptions([&] {
        auto new_ctx = ctx->with_device(device);
        return new compas::CompasContext(new_ctx);
    });
}

extern "C" void compas_destroy_context(const compas::CompasContext* ctx) {
    catch_exceptions([&] { delete ctx; });
}

extern "C" void compas_destroy_object(const Object* obj) {
    catch_exceptions([&] { delete obj; });
}

extern "C" void compas_synchronize(const compas::CompasContext* ctx) {
    catch_exceptions([&] { ctx->synchronize(); });
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
            COMPAS_ERROR("cannot support rank > 3");
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
            return new_object(context->allocate(data_ptr, sizes[0]));
        } else if (rank == 2) {
            return new_object(context->allocate(data_ptr, sizes[0], sizes[1]));
        } else if (rank == 3) {
            return new_object(context->allocate(data_ptr, sizes[0], sizes[1], sizes[2]));
        } else {
            COMPAS_ERROR("cannot support rank > 3");
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
    catch_exceptions([&] { delete array; });
}

compas::View<float> make_view(const float* ptr, int n) {
    return {ptr, {{n}}};
}

extern "C" const Object* compas_make_tissue_parameters(
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
    //    int num_devices = int(context->runtime().info().num_devices());
    //    int chunk_size = kmm::round_up_to_multiple(kmm::div_ceil(nvoxels, num_devices), 32);
    int chunk_size = nvoxels;

    return catch_exceptions([&] {
        return new_object(
            compas::make_tissue_parameters(
                *context,
                nvoxels,
                chunk_size,
                make_view(T1, nvoxels),
                make_view(T2, nvoxels),
                make_view(B1, nvoxels),
                make_view(B0, nvoxels),
                make_view(rho_x, nvoxels),
                make_view(rho_y, nvoxels),
                make_view(x, nvoxels),
                make_view(y, nvoxels),
                make_view(z, nvoxels)));
    });
}

extern "C" Object* compas_make_fisp_sequence(
    const compas::Array<cfloat>* RF_train,
    const compas::Array<cfloat, 2>* sliceprofiles,
    float TR,
    float TE,
    float TW,
    int max_state,
    float TI,
    int undersampling_factor,
    int repetitions) {
    return catch_exceptions([&] {
        return new_object(
            compas::FISPSequence {
                *RF_train,
                *sliceprofiles,
                TR,
                TE,
                TW,
                max_state,
                TI,
                undersampling_factor,
                repetitions});
    });
}

extern "C" Object* compas_make_pssfp_sequence(
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
        return new_object<compas::pSSFPSequence>(
            *RF_train,
            TR,
            *gamma_dt_RF,
            compas::RepetitionData {dt_ex, dt_inv, dt_pr},
            compas::RepetitionData {gamma_dt_GRz_ex, gamma_dt_GRz_inv, gamma_dt_GRz_pr},
            *z);
    });
}

extern "C" Object* compas_make_cartesian_trajectory(
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    compas::Array<cfloat>* k_start,
    float delta_k) {
    return catch_exceptions([&] {
        return new_object<compas::CartesianTrajectory>(
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            delta_k);
    });
}

extern "C" Object* compas_make_spiral_trajectory(
    int nreadouts,
    int samples_per_readout,
    float delta_t,
    compas::Array<cfloat>* k_start,
    compas::Array<cfloat>* delta_k) {
    return catch_exceptions([&] {
        return new_object<compas::SpiralTrajectory>(
            nreadouts,
            samples_per_readout,
            delta_t,
            *k_start,
            *delta_k);
    });
}

extern "C" kmm::ArrayBase* compas_simulate_magnetization_fisp(
    const compas::CompasContext* context,
    const Object* parameters,
    const Object* sequence) {
    return catch_exceptions([&] {
        return new_object(
            compas::simulate_magnetization(
                *context,
                parameters->unwrap<compas::TissueParameters>(),
                sequence->unwrap<compas::FISPSequence>()));
    });
}

extern "C" kmm::ArrayBase* compas_simulate_magnetization_pssfp(
    const compas::CompasContext* context,
    const Object* parameters,
    const Object* sequence) {
    return catch_exceptions([&] {
        return new_object(
            compas::simulate_magnetization(
                *context,
                parameters->unwrap<compas::TissueParameters>(),
                sequence->unwrap<compas::pSSFPSequence>()));
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_cartesian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const Object* parameters,
    const compas::Array<cfloat, 2>* coils,
    const Object* trajectory,
    int32_t low_precision) {
    auto method = low_precision == 0 ? compas::SimulateSignalMethod::MatmulFast : compas::SimulateSignalMethod::MatmulLow;

    return catch_exceptions([&] {
        return new_object(
            compas::magnetization_to_signal(
                *context,
                *echos,
                parameters->unwrap<compas::TissueParameters>(),
                trajectory->unwrap<compas::CartesianTrajectory>(),
                *coils,
                method));
    });
}

extern "C" kmm::ArrayBase* compas_magnetization_to_signal_spiral(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const Object* parameters,
    const compas::Array<cfloat, 2>* coils,
    const Object* trajectory) {
    return catch_exceptions([&] {
        return new_object(
            compas::magnetization_to_signal(
                *context,
                *echos,
                parameters->unwrap<compas::TissueParameters>(),
                trajectory->unwrap<compas::SpiralTrajectory>(),
                *coils));
    });
}

extern "C" compas::Array<cfloat, 3>* compas_compute_jacobian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::Array<cfloat, 2>* delta_echos_T1,
    const compas::Array<cfloat, 2>* delta_echos_T2,
    const Object* parameters,
    const compas::Array<cfloat, 2>* coils,
    const Object* trajectory,
    const compas::Array<cfloat, 2>* vector,
    const int32_t low_precision) {
    auto method = low_precision == 0 ? compas::JacobianComputeMethod::GemmFast : compas::JacobianComputeMethod::GemmLow;

    return catch_exceptions([&] {
        return new_object(
            compas::compute_jacobian(
                *context,
                *echos,
                *delta_echos_T1,
                *delta_echos_T2,
                parameters->unwrap<compas::TissueParameters>(),
                trajectory->unwrap<compas::CartesianTrajectory>(),
                *coils,
                *vector,
                method));
    });
}

extern "C" compas::Array<cfloat, 2>* compas_compute_jacobian_hermitian(
    const compas::CompasContext* context,
    int ncoils,
    const compas::Array<cfloat, 2>* echos,
    const compas::Array<cfloat, 2>* delta_echos_T1,
    const compas::Array<cfloat, 2>* delta_echos_T2,
    const Object* parameters,
    const Object* trajectory,
    const compas::Array<cfloat, 2>* coils,
    const compas::Array<cfloat, 3>* vector,
    const int32_t low_precision) {
    auto method = low_precision == 0 ? compas::JacobianComputeMethod::GemmFast : compas::JacobianComputeMethod::GemmLow;

    return catch_exceptions([&] {
        return new_object(
            compas::compute_jacobian_hermitian(
                *context,
                *echos,
                *delta_echos_T1,
                *delta_echos_T2,
                parameters->unwrap<compas::TissueParameters>(),
                trajectory->unwrap<compas::CartesianTrajectory>(),
                *coils,
                *vector,
                method));
    });
}

extern "C" compas::Array<cfloat, 2>* phase_encoding(
    const compas::CompasContext* context,
    const compas::Array<cfloat, 2>* echos,
    const Object* parameters,
    const Object* trajectory) {
    return catch_exceptions([&] {
        return new_object(
            compas::phase_encoding(
                *context,
                *echos,
                parameters->unwrap<compas::TissueParameters>(),
                trajectory->unwrap<compas::CartesianTrajectory>()));
    });
}

extern "C" compas::Array<cfloat, 3>* compas_compute_residual(
    const compas::CompasContext* context,
    const compas::Array<cfloat, 3>* lhs,
    const compas::Array<cfloat, 3>* rhs,
    float* sum) {
    return catch_exceptions(
        [&] { return new_object(compas::compute_residual(*context, *lhs, *rhs, sum)); });
}

extern "C" compas::Array<cfloat, 2>* compas_simulate_magnetization_derivative_pssfp(
    const compas::CompasContext* context,
    int field,
    const compas::Array<cfloat, 2>* echos,
    const Object* parameters,
    float delta,
    const Object* sequence) {
    return catch_exceptions([&] {
        return new_object(
            compas::simulate_magnetization_derivative(
                *context,
                field,
                *echos,
                parameters->unwrap<compas::TissueParameters>(),
                sequence->unwrap<compas::pSSFPSequence>(),
                delta));
    });
}

extern "C" compas::Array<cfloat, 2>* compas_simulate_magnetization_derivative_fisp(
    const compas::CompasContext* context,
    int field,
    const compas::Array<cfloat, 2>* echos,
    const Object* parameters,
    float delta,
    const Object* sequence) {
    return catch_exceptions([&] {
        return new_object(
            compas::simulate_magnetization_derivative(
                *context,
                field,
                *echos,
                parameters->unwrap<compas::TissueParameters>(),
                sequence->unwrap<compas::FISPSequence>(),
                delta));
    });
}