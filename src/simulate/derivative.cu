#include "compas/core/utils.h"
#include "compas/parameters/tissue.h"
#include "compas/sequences/fisp.h"
#include "compas/sequences/pssfp.h"
#include "compas/simulate/fisp.h"
#include "compas/simulate/pssfp.h"
#include "derivative_kernels.cuh"

namespace compas {

template<typename SequenceView>
void simulate_magnetization_derivative_impl(
    const kmm::DeviceContext& context,
    kmm::Bounds<2> range,
    int field,
    gpu_subview_mut<float, 2> new_parameters,
    gpu_subview_mut<cfloat, 2> delta_echos,
    gpu_subview<cfloat, 2> echos,
    TissueParametersView tissue,
    SequenceView sequence,
    float delta) {
    auto nvoxels = range.x.size();
    auto nreadouts = range.y.size();

    COMPAS_ASSERT(echos.begin(0) == range.y.begin);
    COMPAS_ASSERT(echos.end(0) == range.y.end);
    COMPAS_ASSERT(echos.begin(1) == range.x.begin);
    COMPAS_ASSERT(echos.end(1) == range.x.end);
    COMPAS_ASSERT(field >= 0 && field < TissueParameterField::NUM_FIELDS);

    dim3 block_size = 256;
    dim3 num_blocks = div_ceil(uint(nvoxels), block_size.x);
    compas::kernels::add_difference_to_parameters<<<num_blocks, block_size, 0, context.stream()>>>(
        range.x,
        new_parameters,
        tissue.parameters,
        field,
        delta);

    auto new_tissue = TissueParametersView {tissue};
    new_tissue.parameters = new_parameters;

    simulate_magnetization_kernel(context, range[0], delta_echos, new_tissue, sequence);

    num_blocks = {div_ceil(uint(nvoxels), block_size.x), div_ceil(uint(nreadouts), block_size.y)};
    compas::kernels::calculate_finite_difference<<<num_blocks, block_size, 0, context.stream()>>>(
        range,
        delta_echos,
        echos,
        1.0F / delta);
}

Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    pSSFPSequence sequence,
    float delta) {
    using namespace kmm::placeholders;
    auto nvoxels = parameters.nvoxels;
    auto nreadouts = sequence.RF_train.size();
    auto chunk_size = parameters.chunk_size;

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    auto new_parameters = Array<float, 2> {parameters.data.sizes()};
    auto delta_echos = Array<cfloat, 2> {echos.sizes()};

    context.parallel_device(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        simulate_magnetization_derivative_impl<pSSFPSequenceView>,
        _xy,
        field,
        write(new_parameters(_, _x)),
        write(delta_echos(_, _x)),
        echos(_, _x),
        read(parameters, _x),
        sequence,
        delta);

    return delta_echos;
}

Array<cfloat, 2> simulate_magnetization_derivative(
    const CompasContext& context,
    int field,
    Array<cfloat, 2> echos,
    TissueParameters parameters,
    FISPSequence sequence,
    float delta) {
    using namespace kmm::placeholders;
    auto nvoxels = parameters.nvoxels;
    auto nreadouts = sequence.RF_train.size();
    auto chunk_size = parameters.chunk_size;

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    auto new_parameters = Array<float, 2> {parameters.data.sizes()};
    auto delta_echos = Array<cfloat, 2> {echos.sizes()};

    context.parallel_device(
        {nvoxels, nreadouts},
        {chunk_size, nreadouts},
        simulate_magnetization_derivative_impl<FISPSequenceView>,
        _xy,
        field,
        write(new_parameters(_, _x)),
        write(delta_echos(_, _x)),
        echos(_, _x),
        read(parameters, _x),
        sequence,
        delta);

    return delta_echos;
}

}  // namespace compas
