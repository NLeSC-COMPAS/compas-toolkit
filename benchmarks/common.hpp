#pragma once

#include <random>
#include <iostream>
#include <chrono>
#include "compas/parameters/tissue.h"
#include "compas/core/context.h"

template <typename F>
static std::pair<double, int> benchmark(F fun) {
    auto before = std::chrono::high_resolution_clock::now();
    auto deadline = before + std::chrono::seconds(1);
    auto after = before;
    int runs = 0;

    while (after < deadline) {
        fun();
        after = std::chrono::high_resolution_clock::now();
        runs++;
    }

    double duration =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count()) * 1e-6 / runs;

    return {duration, runs};
}

static compas::TissueParameters generate_tissue_parameters(const compas::CompasContext& context, int nvoxels) {
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
    for (size_t i = 0; i < size_t(nvoxels); i++) {
        T1[i] = 0.85f;
        T2[i] = 0.05f;
        B1[i] = 1.0f;
        B0[i] = 0.0f;
        rho_x[i] = 3.0f;
        rho_y[i] = 4.0f;
        x[i] = float(i / width) * 25.6f;
        y[i] = float(i % width) * 25.6f;
        z[i] = 0.0f;
    }

    auto parameters = make_tissue_parameters(
        context,
        nvoxels,
        nvoxels,
        {T1.data(), {{nvoxels}}},
        {T2.data(), {{nvoxels}}},
        {B1.data(), {{nvoxels}}},
        {B0.data(), {{nvoxels}}},
        {rho_x.data(), {{nvoxels}}},
        {rho_y.data(), {{nvoxels}}},
        {x.data(), {{nvoxels}}},
        {y.data(), {{nvoxels}}},
        {z.data(), {{nvoxels}}});

    return parameters;
}
