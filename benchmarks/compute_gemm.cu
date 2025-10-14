#include "common.hpp"
#include "compas/utils/gemm.h"

#define LAMBDA(f) [](auto&&... xs) { return f(::std::forward<decltype(xs)>(xs)...); }

using namespace compas;

template <typename ComplexT>
static void benchmark_method(
        std::string name,
        const CompasContext& context,
        Array<cfloat, 2> lhs,
        Array<cfloat, 2> rhs,
        GemmComputeMethod method,
        std::vector<cfloat> expected_result) {
    auto result = Array<cfloat, 2>{{lhs.size(0), rhs.size(0)}};
    auto result_planar = Array<float, 3>{{2, lhs.size(0), rhs.size(0)}};

    auto lhs_planar = Array<ComplexT, 3> {{2, lhs.size(0), lhs.size(1)}};
    auto rhs_planar = Array<ComplexT, 3> {{2, rhs.size(0), rhs.size(1)}};

    context.submit_device(
            LAMBDA(convert_complex_to_planar),
            write(lhs_planar),
            lhs
    );

    context.submit_device(
            LAMBDA(convert_complex_to_planar),
            write(rhs_planar),
            rhs
    );

    context.synchronize();

    auto [duration, runs] = benchmark([&] {
        context.submit_device(
                LAMBDA(compute_complex_gemm),
                write(result_planar),
                lhs_planar,
                rhs_planar,
                1.0f,
                0.0f,
                method
        );

        context.synchronize();
    });

    context.submit_device(
            convert_planar_to_complex,
            write(result),
            result_planar
    );

    context.synchronize();

    std::cout << "benchmark: " << name << "\n";
    std::cout << "iterations: " << runs << "\n";
    std::cout << "time: " << duration << " milliseconds\n";
    compare_output(expected_result, result.copy_to_vector());
    std::cout << "\n";
}

std::vector<cfloat> compute_gemm_naive(
        int N,
        int M,
        int K,
        const std::vector<cfloat>& lhs,
        const std::vector<cfloat>& rhs
) {
    static constexpr int TILE_SIZE_I = 8;
    static constexpr int TILE_SIZE_J = 8;
    std::vector<cdouble> result (N * M, cdouble());

#pragma omp parallel for collapse(2) schedule(static)
    for (int base_i = 0; base_i < N; base_i += TILE_SIZE_I) {
        for (int base_j = 0; base_j < M; base_j += TILE_SIZE_J) {
            cdouble result_tile[TILE_SIZE_I][TILE_SIZE_J];

            for (int di = 0; di < TILE_SIZE_I; di++) {
                for (int dj = 0; dj < TILE_SIZE_J; dj++) {
                    result_tile[di][dj] = 0;
                }
            }

            for (int k = 0; k < K; k++) {
                cdouble lhs_tile[TILE_SIZE_I];
                cdouble rhs_tile[TILE_SIZE_J];

                for (int di = 0; di < TILE_SIZE_I; di++) {
                    int i = base_i + di;
                    lhs_tile[di] = cdouble(i < N && k < K ? lhs[i * K + k] : 0);
                }

                for (int dj = 0; dj < TILE_SIZE_J; dj++) {
                    int j = base_j + dj;
                    rhs_tile[dj] = cdouble(j < M && k < K ? rhs[j * K + k] : 0);
                }

                for (int di = 0; di < TILE_SIZE_I; di++) {
                    for (int dj = 0; dj < TILE_SIZE_J; dj++) {
                        result_tile[di][dj] += lhs_tile[di] * rhs_tile[dj];
                    }
                }
            }

            for (int di = 0; di < TILE_SIZE_I; di++) {
                for (int dj = 0; dj < TILE_SIZE_J; dj++) {
                    int i = base_i + di;
                    int j = base_j + dj;

                    if (i < N && j < M) {
                        result[i * M + j] = result_tile[di][dj];
                    }
                }
            }
        }
    }

    std::vector<cfloat> result_lo (result.size(), cfloat());

    for (size_t i = 0; i < result.size(); i++) {
        result_lo[i] = cfloat(result[i]);
    }

    return result_lo;
}

int main() {
    auto context = make_context();

    int n = 1120;
    int m = 224;
    int k = 50176;

    std::cout << "parameters:\n";
    std::cout << "N: " << n << "\n";
    std::cout << "M: " << m << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "\n";

    auto lhs = generate_random_complex(context, n, k);
    auto rhs = generate_random_complex(context, m, k);

    auto expected = compute_gemm_naive(
        n,m,k, lhs.copy_to_vector(), rhs.copy_to_vector()
    );

    // float
    benchmark_method<float>(
            "matmul F32 (pedantic)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Pedantic,
            expected);

    benchmark_method<float>(
            "matmul F32 (regular)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Regular,
            expected);

    benchmark_method<float>(
            "matmul F32 (fast)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Fast,
            expected);

    // bfloat16
    benchmark_method<kernel_float::bfloat16_t>(
            "matmul BF16 (pedantic)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Pedantic,
            expected);

    benchmark_method<kernel_float::bfloat16_t>(
            "matmul BF16 (regular)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Regular,
            expected);

    benchmark_method<kernel_float::bfloat16_t>(
            "matmul BF16 (fast)",
            context,
            lhs,
            rhs,
            GemmComputeMethod::Fast,
            expected);

    return 0;
}
