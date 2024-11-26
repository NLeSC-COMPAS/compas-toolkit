#include <chrono>
#include <iostream>
#include <random>

#include "common.hpp"
#include "compas/core/complex_type.h"
#include "compas/core/context.h"
#include "compas/sequences/fisp.h"
#include "compas/simulate/fisp.h"

using namespace compas;

int main() {
    auto context = make_context();
    int nvoxels = 256 * 256;
    int nreadouts = 256;
    float TR = 0.010f;
    float TE = 0.005f;
    float TI = 0.100f;
    int max_state = 25;
    int num_slices = 35;

    auto echos = Array<cfloat, 2> {{nreadouts, nvoxels}};

    auto RF_train = std::vector<cfloat>(size_t(nreadouts));
    for (size_t i = 0; i < RF_train.size(); i++) {
        RF_train[i] = float(i * 90) / float(nreadouts - 1);
    }

    auto sliceprofiles = std::vector<cfloat>(size_t(nreadouts * num_slices));
    for (size_t i = 0; i < sliceprofiles.size(); i++) {
        sliceprofiles[i] = 1;
    }

    auto parameters = generate_tissue_parameters(context, nvoxels);
    auto sequence = make_fisp_sequence(
        context,
        host_view<cfloat> {RF_train.data(), {{nreadouts}}},
        host_view<cfloat, 2> {sliceprofiles.data(), {{num_slices, nreadouts}}},
        TR,
        TE,
        max_state,
        TI);

    auto states = std::vector<int> {5, 10, 16, 20, 25, 30, 32, 35, 50};
    states = {25};

    for (int max_state : states) {
        sequence.max_state = max_state;

        auto [duration, runs] = benchmark([&] {
            echos = compas::simulate_magnetization(context, parameters, sequence);

            context.synchronize();
        });

        std::cout << "benchmark: "
                  << "FISP with no. of states: " << max_state << "\n";
        std::cout << "iterations: " << runs << "\n";
        std::cout << "time: " << duration << " milliseconds\n";
        std::cout << "\n";
        break;
    }

    return EXIT_SUCCESS;
}