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
    kmm::NDRange,
    int field,
    gpu_view_mut<float, 2> new_parameters,
    gpu_view_mut<cfloat, 2> delta_echos,
    gpu_view<cfloat, 2> echos,
    TissueParametersView tissue,
    SequenceView sequence,
    float delta) {
    auto nvoxels = tissue.nvoxels;
    auto nreadouts = kmm::checked_cast<int>(sequence.RF_train.size());

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);
    COMPAS_ASSERT(field >= 0 && field < TissueParameterField::NUM_FIELDS);

    dim3 block_size = 256;
    dim3 num_blocks = div_ceil(uint(nvoxels), block_size.x);
    compas::kernels::add_difference_to_parameters<<<num_blocks, block_size, 0, context.stream()>>>(
        nvoxels,
        new_parameters,
        tissue.parameters,
        field,
        delta);

    auto new_tissue = TissueParametersView {
        .parameters = new_parameters,
        .nvoxels = nvoxels,
        .has_z = tissue.has_z,
        .has_b0 = tissue.has_b0,
        .has_b1 = tissue.has_b1,
    };

    simulate_magnetization_kernel(context, delta_echos, new_tissue, sequence);

    num_blocks = {div_ceil(uint(nvoxels), block_size.x), div_ceil(uint(nreadouts), block_size.y)};

    compas::kernels::calculate_finite_difference<<<num_blocks, block_size, 0, context.stream()>>>(
        nreadouts,
        nvoxels,
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
    auto nvoxels = parameters.nvoxels;
    auto nreadouts = sequence.RF_train.size();

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    auto new_parameters = Array<float, 2> {parameters.parameters.sizes()};
    auto delta_echos = Array<cfloat, 2> {echos.sizes()};

    context.submit_device(
        {nreadouts, nvoxels},
        simulate_magnetization_derivative_impl<pSSFPSequenceView>,
        field,
        write(new_parameters),
        write(delta_echos),
        echos,
        parameters,
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
    auto nvoxels = parameters.nvoxels;
    auto nreadouts = sequence.RF_train.size();

    COMPAS_ASSERT(echos.size(0) == nreadouts);
    COMPAS_ASSERT(echos.size(1) == nvoxels);

    auto new_parameters = Array<float, 2> {parameters.parameters.sizes()};
    auto delta_echos = Array<cfloat, 2> {echos.sizes()};

    context.submit_device(
        {nreadouts, nvoxels},
        simulate_magnetization_derivative_impl<FISPSequenceView>,
        field,
        write(new_parameters),
        write(delta_echos),
        echos,
        parameters,
        sequence,
        delta);

    return delta_echos;
}

}  // namespace compas
