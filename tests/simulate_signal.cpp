#include "catch2/catch_all.hpp"
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

    for (int i = 0; i < nvoxels; i++) {
        T1[i] = 1;
        T2[i] = 2;
        B1[i] = 1;
        B0[i] = 0;
        rho_x[i] = 0;
        rho_y[i] = 0;
        x[i] = float(i);
        y[i] = 0;
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

TEST_CASE("simulate_signal_cartesian") {
    auto context = make_context();

    int ncoils = 2;
    int nvoxels = 256;
    int nreadouts = 16;
    int samples_per_readout = 16;
    float delta_t = 1;
    float delta_k = 1;

    auto h_echos = std::vector<cfloat>(nreadouts * nvoxels);
    for (int i = 0; i < nreadouts * nvoxels; i++) {
        h_echos[i] = cfloat(i / h_echos.size());
    }

    auto signal = context.allocate<cfloat, 3>({ncoils, nreadouts, samples_per_readout});
    auto echos = context.allocate<cfloat, 2>({nreadouts, nvoxels});
    auto coil_sensitivities = context.allocate<float, 2>({ncoils, nvoxels});

    echos.copy_from(h_echos);

    TissueParameters parameters = generate_tissue_parameters(context, nvoxels);

    auto k_start = std::vector<cfloat>(nreadouts);
    auto trajectory = make_cartesian_trajectory(
        context,
        nreadouts,
        samples_per_readout,
        delta_t,
        {k_start.data(), {nreadouts}},
        delta_k);

    simulate_signal(context, signal, echos, parameters, trajectory, coil_sensitivities);
}
