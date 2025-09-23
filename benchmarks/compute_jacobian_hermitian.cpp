#include <chrono>
#include <iostream>
#include <random>

#include "common.hpp"
#include "compas/jacobian/hermitian.h"
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
    Array<cfloat, 3> vector,
    const std::vector<cfloat>& expected) {
    Array<cfloat, 2> JHv;
    context.synchronize();

    auto [duration, runs] = benchmark([&] {
        JHv = compute_jacobian_hermitian(
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
    compare_output(expected, JHv.copy_to_vector());
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
    auto vector = generate_random_complex(context, ncoils, nreadouts, samples_per_readout);
    auto coil_sensitivities = generate_random_complex(context, ncoils, nvoxels);
    TissueParameters parameters = generate_tissue_parameters(context, nvoxels);

    float delta_t = 10e-5;
    float delta_k = 2 * M_PI / 25.6;
    auto k_start = std::vector<cfloat>(nreadouts);

    auto trajectory = make_cartesian_trajectory(
        context,
        nreadouts,
        samples_per_readout,
        delta_t,
        compas::View<cfloat> {k_start.data(), {{nreadouts}}},
        delta_k);

    auto JHv = compas::compute_jacobian_hermitian(
        context,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        parameters,
        trajectory,
        coil_sensitivities,
        vector,
        JacobianComputeMethod::Naive);

    auto JHv_ref = JHv.copy_to_vector();

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
        JHv_ref);

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
        JHv_ref);

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
        JHv_ref);

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
        JHv_ref);

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
        JHv_ref);

    return 0;
}
