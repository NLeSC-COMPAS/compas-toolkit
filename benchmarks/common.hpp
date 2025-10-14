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

    // One call to warm up
    fun();

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

template <typename T, typename F>
compas::Array<T, 2> generate_input(
        compas::CompasContext& context,
        F generator,
        int height,
        int width
) {
    std::vector<T> h_data(height * width);

    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            h_data[i * width + j] = generator(i, j);
        }
    }

    return context.allocate(h_data.data(), height, width);
}

template <typename T, typename F>
compas::Array<T, 3> generate_input(
        compas::CompasContext& context,
        F generator,
        int height,
        int width,
        int depth
) {
    std::vector<T> h_data(height * width * depth);

    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            for (int64_t k = 0; k < depth; k++) {
                h_data[(i * width + j) * depth + k] = generator(i, j, k);
            }
        }
    }

    return context.allocate(h_data.data(), height, width, depth);
}

static compas::Array<compas::cfloat, 2> generate_random_complex(
        compas::CompasContext& context,
        int height,
        int width
) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return generate_input<compas::cfloat>(
            context,
            [&](int i, int j) {
                return compas::cfloat {dist(gen), dist(gen)};
                // return compas::polar(dist(gen), dist(gen) * 2.0F * float(M_PI));
            },
            height,
            width
    );
}

static compas::Array<compas::cfloat, 3> generate_random_complex(
        compas::CompasContext& context,
        int height,
        int width,
        int depth
) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return generate_input<compas::cfloat>(
            context,
            [&](int i, int j, int k) {
                return compas::polar(
                        dist(gen),
                        dist(gen) * 2.0f * float(M_PI)
                );
            },
            height,
            width,
            depth
    );
}

template <typename T>
static double compute_within_error(
        const std::vector<T>& expected,
        const std::vector<T>& answer,
        double rtol,
        double atol=1e-9
) {
    size_t n = 0;

    for (size_t i = 0; i < expected.size(); i++) {
        double err = abs(answer[i] - expected[i]);

        if (err <= atol || err <= rtol * abs(expected[i])) {
            n++;
        }
    }

    auto f = double(n) / double(expected.size());
    return round(f * 10000) / 10000;
}

template <typename T>
static void compare_output(
        const std::vector<T>& expected,
        const std::vector<T>& answer
) {
    double max_abs_error = 0;
    double max_rel_error = 0;
    double total_abs_error = 0;
    double total_rel_error = 0;

    for (size_t i = 0; i < expected.size(); i++) {
        double abs_error = abs((answer[i] - expected[i]));
        double rel_error = abs_error / std::max(1e-10f, abs(expected[i]));

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);

        total_abs_error += abs_error;
        total_rel_error += rel_error;
    }

    std::cout << "average absolute error: " << (total_abs_error / expected.size()) << "\n";
    std::cout << "average relative error: " << (total_rel_error / expected.size()) << "\n";
    std::cout << "maximum absolute error: " << max_abs_error << "\n";
    std::cout << "maximum relative error: " << max_rel_error << "\n";
    std::cout << "fraction within 1%: " << compute_within_error(expected, answer, 0.01) * 100 << "%\n";
    std::cout << "fraction within 0.1%: " << compute_within_error(expected, answer, 0.001) * 100 << "%\n";
    std::cout << "fraction within 0.01%: " << compute_within_error(expected, answer, 0.0001) * 100 << "%\n";
}