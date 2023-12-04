#pragma once

#include "trajectories/cartesian.h"
#include "simulate/signal.h"
#include "parameters/tissue.h"
#include "core/context.h"
#include "core/complex_type.h"
#include <random>
#include <iostream>
#include <chrono>

template <typename F>
static std::pair<double, int> benchmark(F fun) {
    auto before = std::chrono::high_resolution_clock::now();
    auto after = before;
    int runs = 0;

    do {
        fun();
        after = std::chrono::high_resolution_clock::now();
        runs++;
    } while (after < before + std::chrono::seconds(1));

    double duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count() * 1e-6 / runs;

    return {duration, runs};
}

static compas::TissueParameters generate_tissue_parameters(const compas::CudaContext& context, int nvoxels) {
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
