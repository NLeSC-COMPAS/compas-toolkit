#include <chrono>
#include <iostream>
#include <random>

#include "common.hpp"
#include "compas/jacobian/product.h"
#include "compas/trajectories/cartesian.h"
#include "compas/trajectories/signal.h"

using namespace compas;

static void benchmark_method(
    std::string name,
    JacobianComputeMethod kind,
    const CompasContext& context,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> delta_echos_T1,
    Array<cfloat, 2> delta_echos_T2,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    Array<cfloat, 2> coil_sensitivities,
    Array<cfloat, 2> vector,
    const std::vector<cfloat>& expected_Jv) {
    Array<cfloat, 3> Jv;
    context.synchronize();

    auto [duration, runs] = benchmark([&] {
        Jv = compute_jacobian(
            context,
            echos,
            delta_echos_T1,
            delta_echos_T2,
            parameters,
            trajectory,
            coil_sensitivities,
            vector,
            kind);

        context.synchronize();
    });

    context.synchronize();

    std::cout << "benchmark: " << name << "\n";
    std::cout << "iterations: " << runs << "\n";
    std::cout << "time: " << duration << " milliseconds\n";
    compare_output(expected_Jv, Jv.copy_to_vector());
    std::cout << "\n";
}

int main() {
    auto context = make_context();

    int ncoils = 1;
    int nvoxels = 50176;
    int nreadouts = 1120;
    int samples_per_readout = 224;

    std::cout << "parameters:\n";
    std::cout << "coils: " << ncoils << "\n";
    std::cout << "voxels: " << nvoxels << "\n";
    std::cout << "readouts: " << nreadouts << "\n";
    std::cout << "samples per readout: " << samples_per_readout << "\n";
    std::cout << "\n";

    auto echos = generate_random_complex(context, nreadouts, nvoxels);
    auto delta_echos_T1 = generate_random_complex(context, nreadouts, nvoxels);
    auto delta_echos_T2 = generate_random_complex(context, nreadouts, nvoxels);
    auto vector = generate_random_complex(context, 4, nvoxels);
    auto coil_sensitivities = generate_random_complex(context, ncoils, nvoxels);
    TissueParameters parameters = generate_tissue_parameters(context, nvoxels);

    auto delta_t = 10e-5f;
    auto delta_k = 2.0f * float(M_PI) / 25.6f;
    auto k_start = std::vector<cfloat>(size_t(nreadouts));

    auto trajectory = make_cartesian_trajectory(
        context,
        nreadouts,
        samples_per_readout,
        delta_t,
        compas::View<cfloat> {k_start.data(), {{nreadouts}}},
        delta_k);

    auto Jv = compas::compute_jacobian(
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        JacobianComputeMethod::Naive);

    auto Jv_ref = Jv.copy_to_vector();

    benchmark_method(
        "naive",
        JacobianComputeMethod::Naive,
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        Jv_ref);

    benchmark_method(
        "direct",
        JacobianComputeMethod::Direct,
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        Jv_ref);

    benchmark_method(
        "matmul",
        JacobianComputeMethod::Gemm,
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        Jv_ref);

    benchmark_method(
        "matmul (faster)",
        JacobianComputeMethod::GemmFast,
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        Jv_ref);

    benchmark_method(
        "matmul (low precision)",
        JacobianComputeMethod::GemmLow,
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        Jv_ref);

    return 0;
}
