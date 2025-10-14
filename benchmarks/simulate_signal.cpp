#include <chrono>
#include <iostream>
#include <random>

#include "common.hpp"
#include "compas/trajectories/cartesian.h"
#include "compas/trajectories/signal.h"

using namespace compas;

static void benchmark_method(
    std::string name,
    SimulateSignalMethod method,
    const CompasContext& context,
    Array<cfloat, 2> echos,
    Array<cfloat, 2> coil_sensitivities,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    const std::vector<cfloat>& signal_ref) {
    Array<cfloat, 3> signal;
    context.synchronize();

    auto [duration, runs] = benchmark([&] {
        signal = compas::magnetization_to_signal(
            context,
            echos,
            parameters,
            trajectory,
            coil_sensitivities,
            method);

        context.synchronize();
    });

    context.synchronize();

    std::cout << "benchmark: " << name << "\n";
    std::cout << "iterations: " << runs << "\n";
    std::cout << "time: " << duration << " milliseconds\n";
    compare_output(signal_ref, signal.copy_to_vector());
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

    auto signal = compas::magnetization_to_signal(
        context,
        echos,
        parameters,
        trajectory,
        coil_sensitivities,
        SimulateSignalMethod::Naive);

    auto signal_ref = signal.copy_to_vector();

    benchmark_method(
        "naive",
        SimulateSignalMethod::Naive,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "direct",
        SimulateSignalMethod::Direct,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "matmul (pedantic)",
        SimulateSignalMethod::MatmulPedantic,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "matmul (regular)",
        SimulateSignalMethod::Matmul,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "matmul (faster)",
        SimulateSignalMethod::MatmulFast,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "matmul (low precision)",
        SimulateSignalMethod::MatmulLow,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    return 0;
}
