#include "compas/core/utils.h"
#include "compas/parameters/tissue.h"
#include "compas/sequences/fisp.h"
#include "compas/sequences/pssfp.h"
#include "compas/simulate/fisp.h"
#include "compas/simulate/pssfp.h"
#include "derivative_kernels.cuh"

namespace compas {

template<typename Sequence>
Array<cfloat, 2> simulate_magnetization_derivative_impl(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    Sequence sequence,
    float delta) {
    using namespace kmm::placeholders;
    auto nvoxels = parameters.nvoxels;
    auto chunk_size = parameters.chunk_size;
    auto nreadouts = echos.size(0);

    COMPAS_CHECK(echos.size(1) == nvoxels);
    COMPAS_CHECK(field >= 0 && field < TissueParameterField::NUM_FIELDS);

    auto new_data = Array<float, 2> {parameters.data.size()};

    // Add difference to parametrs
    context.parallel_kernel(
        nvoxels,
        chunk_size,
        256,
        compas::kernels::add_difference_to_parameters,
        _x,
        write(new_data[_][_x]),
        parameters.data[_][_x],
        field,
        delta);

    // Create new tissue parameters
    auto new_parameters = parameters;
    new_parameters.data = new_data;

    // Compute magnetization at `f(parameters + delta)`
    auto next_echos = simulate_magnetization(context, TissueParameters(new_parameters), sequence);

    // Compute `delta_echos = (echos - next_echos) / delta`
    auto delta_echos = Array<cfloat, 2> {echos.size()};
    context.parallel_kernel(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        256,
        compas::kernels::calculate_finite_difference,
        _xy,
        write(delta_echos[_][_x]),
        echos[_][_x],
        next_echos[_][_x],
        1.0F / delta);

    return delta_echos;
}

Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence,
    float delta) {
    return simulate_magnetization_derivative_impl(
        context,
        field,
        echos,
        parameters,
        sequence,
        delta);
}

Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    FISPSequence sequence,
    float delta) {
    return simulate_magnetization_derivative_impl(
        context,
        field,
        echos,
        parameters,
        sequence,
        delta);
}

}  // namespace compas
