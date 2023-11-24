#include <chrono>
#include <iostream>
#include <random>

#include "core/complex_type.h"
#include "core/context.h"
#include "parameters/tissue.h"
#include "simulate/signal.h"
#include "trajectories/cartesian.h"

using namespace compas;

static TissueParameters generate_tissue_parameters(const CudaContext& context, int nvoxels) {
    auto T1 = std::vector<float>(nvoxels);
    auto T2 = std::vector<float>(nvoxels);
    auto B1 = std::vector<float>(nvoxels);
    auto B0 = std::vector<float>(nvoxels);
    auto rho_x = std::vector<float>(nvoxels);
    auto rho_y = std::vector<float>(nvoxels);
    auto x = std::vector<float>(nvoxels);
    auto y = std::vector<float>(nvoxels);
    auto z = std::vector<float>(nvoxels);

    int width = int(sqrt(nvoxels));
    for (int i = 0; i < nvoxels; i++) {
        T1[i] = 0.85;
        T2[i] = 0.05;
        B1[i] = 1;
        B0[i] = 0;
        rho_x[i] = 3;
        rho_y[i] = 4;
        x[i] = float(i / width) * 25.6;
        y[i] = float(i % width) * 25.6;
        z[i] = 0;
    }

    auto parameters = make_tissue_parameters(
        context,
        nvoxels,
        {T1.data(), {nvoxels}},
        {T2.data(), {nvoxels}},
        {B1.data(), {nvoxels}},
        {B0.data(), {nvoxels}},
        {rho_x.data(), {nvoxels}},
        {rho_y.data(), {nvoxels}},
        {x.data(), {nvoxels}},
        {y.data(), {nvoxels}},
        {z.data(), {nvoxels}});

    return parameters;
}

void benchmark_method(
    std::string name,
    SimulateSignalMethod method,
    const CudaContext& context,
    CudaArray<cfloat, 3> signal,
    CudaArray<cfloat, 2> echos,
    CudaArray<float, 2> coil_sensitivities,
    TissueParameters parameters,
    CartesianTrajectory trajectory,
    std::vector<cfloat>& signal_ref) {
    auto before = std::chrono::high_resolution_clock::now();
    auto after = before;
    int runs = 0;

    do {
        simulate_signal_cartesian(
            context,
            signal.view_mut(),
            echos.view(),
            parameters.view(),
            trajectory.view(),
            coil_sensitivities.view(),
            method);

        after = std::chrono::high_resolution_clock::now();
        runs++;
    } while (after < before + std::chrono::seconds(1));

    auto signal_result = std::vector<cfloat>(signal.size());
    signal.copy_to(signal_result);

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

    double duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count() * 1e-6 / runs;

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

    auto h_coils = std::vector<float>(ncoils * nvoxels);
    for (int i = 0; i < ncoils; i++) {
        std::fill_n(h_coils.data() + nvoxels * i, nvoxels, float(i + 1) / ncoils);
    }

    auto signal = context.allocate<cfloat, 3>({ncoils, nreadouts, samples_per_readout});
    auto echos = context.allocate<cfloat, 2>({nreadouts, nvoxels});
    auto coil_sensitivities = context.allocate<float, 2>({ncoils, nvoxels});

    echos.copy_from(h_echos);
    coil_sensitivities.copy_from(h_coils);

    TissueParameters parameters = generate_tissue_parameters(context, nvoxels);

    float delta_t = 10e-5;
    float delta_k = 2 * M_PI / 25.6;
    auto k_start = std::vector<cfloat>(nreadouts);

    auto trajectory = make_cartesian_trajectory(
        context,
        nreadouts,
        samples_per_readout,
        delta_t,
        {k_start.data(), {nreadouts}},
        delta_k);

    simulate_signal_cartesian(
        context,
        signal.view_mut(),
        echos.view(),
        parameters.view(),
        trajectory.view(),
        coil_sensitivities.view());

    auto signal_ref = std::vector<cfloat>(signal.size());
    signal.copy_to(signal_ref);

    benchmark_method(
        "direct",
        SimulateSignalMethod::Direct,
        context,
        signal,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);
    benchmark_method(
        "matmul (pedantic)",
        SimulateSignalMethod::MatmulPedantic,
        context,
        signal,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);
    benchmark_method(
        "matmul (regular)",
        SimulateSignalMethod::Matmul,
        context,
        signal,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);
    benchmark_method(
        "matmul (TF32)",
        SimulateSignalMethod::MatmulTF32,
        context,
        signal,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);
    benchmark_method(
        "matmul (BF16)",
        SimulateSignalMethod::MatmulBF16,
        context,
        signal,
        echos,
        coil_sensitivities,
        parameters,
        trajectory,
        signal_ref);

    return 0;
}
