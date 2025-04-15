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
    std::vector<cfloat>& signal_ref) {
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

    std::vector<cfloat> signal_result(signal.size());
    signal.copy_to(signal_result.data(), signal_result.size());

    double max_abs_error = 0;
    double max_rel_error = 0;
    double total_abs_error = 0;
    double total_rel_error = 0;

    for (size_t i = 0; i < signal_ref.size(); i++) {
        double abs_error = abs((signal_result[i] - signal_ref[i]));
        double rel_error = abs_error / std::max(1e-10f, abs(signal_ref[i]));

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);

        total_abs_error += abs_error;
        total_rel_error += rel_error;
    }

    std::cout << "benchmark: " << name << "\n";
    std::cout << "iterations: " << runs << "\n";
    std::cout << "time: " << duration << " milliseconds\n";
    std::cout << "average absolute error: " << (total_abs_error / signal.size()) << "\n";
    std::cout << "average relative error: " << (total_rel_error / signal.size()) << "\n";
    std::cout << "maximum absolute error: " << max_abs_error << "\n";
    std::cout << "maximum relative error: " << max_rel_error << "\n";
    std::cout << "\n";
}

int main() {
    auto context = make_context();

    int ncoils = 4;
    int nvoxels = 256 * 256;
    int nreadouts = 256;
    int samples_per_readout = 256;

    auto h_echos = std::vector<cfloat>(nreadouts * nvoxels);
    for (int i = 0; i < nreadouts * nvoxels; i++) {
        h_echos[i] = float(i) / float(h_echos.size());
    }

    auto h_coils = std::vector<cfloat>(ncoils * nvoxels);
    for (int i = 0; i < ncoils; i++) {
        std::fill_n(h_coils.data() + nvoxels * i, nvoxels, float(i + 1) / ncoils);
    }

    auto echos = context.allocate(h_echos.data(), nreadouts, nvoxels);
    auto coil_sensitivities = context.allocate(h_coils.data(), ncoils, nvoxels);

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

    auto signal_ref = std::vector<cfloat>(signal.size());
    signal.copy_to(signal_ref.data(), signal_ref.size());

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
        "matmul (TF32)",
        SimulateSignalMethod::MatmulTF32,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    benchmark_method(
        "matmul (BF16)",
        SimulateSignalMethod::MatmulBF16,
        context,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    return 0;
}
