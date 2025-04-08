module CompasToolkit
include("Constants.jl")

function version()::String
    ptr = @ccall LIBRARY.compas_version()::Cstring
    return unsafe_string(ptr)
end

function check_version()
    expected_version = VERSION
    gotten_version = version()

    if expected_version != gotten_version
        throw(
            ErrorException(
                "version of `$LIBRARY` is invalid: got version $gotten_version, expected $expected_version",
            ),
        )
    end
end

function unsafe_destroy_object!(obj)
    @ccall LIBRARY.compas_destroy_object(obj.ptr::Ptr{Cvoid})::Cvoid
end

mutable struct Context
    ptr::Ptr{Cvoid}

    function Context(device::Integer)
        check_version()
        ptr = @ccall LIBRARY.compas_make_context(device::Int32)::Ptr{Cvoid}
        obj = new(ptr)
        destroy = (obj) -> @ccall LIBRARY.compas_destroy_context(ptr::Ptr{Cvoid})::Cvoid
        finalizer(destroy, obj)
    end

    Context() = Context(0)
end

Base.unsafe_convert(T::Type{Ptr{Cvoid}}, x::Context) = x.ptr

function init_context(device::Integer)::Context
    c = Context(device)
    set_context(c)
    return c
end

const TASK_LOCAL_STORAGE_KEY::Symbol = :compas_toolkit_global_context

function set_context(context::Context)
    task_local_storage(TASK_LOCAL_STORAGE_KEY, context)
end

function get_context()::Context
    try
        return task_local_storage(TASK_LOCAL_STORAGE_KEY)
    catch e
        throw(ArgumentError("compas toolkit has not been initialized, use `init_context` before usage"))
    end
end

"""
Waits until all asynchronous operations of Compas have finished.
"""
function synchronize()
    context = get_context()
    @ccall LIBRARY.compas_synchronize(context::Ptr{Cvoid})::Cvoid
end

"""
Object representing an array of dimensionality `N` and type `T`.
"""
mutable struct CompasArray{T, N} <: AbstractArray{T, N}
    context::Context
    ptr::Ptr{Cvoid}
    sizes::Dims{N}

    function CompasArray{T, N}(context::Context, ptr::Ptr{Cvoid}, sizes::Dims{N}) where {T, N}
        obj = new(context, ptr, sizes)
        destroy = (obj) -> @ccall LIBRARY.compas_destroy_array(ptr::Ptr{Cvoid})::Cvoid
        finalizer(destroy, obj)
    end
end

# AbstractArray
Base.size(array::CompasArray) = reverse(array.sizes)
Base.getindex(array::CompasArray, i) = throw(ArgumentError("cannot index into a 'CompasArray'"))
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::CompasArray) = x.ptr

# 
Base.show(io::IO, ca::CompasArray) = print(io, typeof(ca), " of size ", size(ca))
Base.show(io::IO, ::MIME"text/plain", ca::CompasArray) = print(io, ca)

function make_array(context::Context, input::Array{Float32, N})::CompasArray{Float32, N} where {N}
    sizes::Vector{Int64} = [reverse(size(input))...]

    ptr = @ccall LIBRARY.compas_make_array_float(
        context::Ptr{Cvoid},
        input::Ptr{Float32},
        N::Int32,
        sizes::Ptr{Int64}
    )::Ptr{Cvoid}

    return CompasArray{Float32, N}(context, ptr, Dims{N}(sizes))
end

function make_array(context::Context, input::Array{ComplexF32, N})::CompasArray{ComplexF32, N} where {N}
    sizes::Vector{Int64} = [reverse(size(input))...]

    ptr = @ccall LIBRARY.compas_make_array_complex(
        context::Ptr{Cvoid},
        input::Ptr{ComplexF32},
        N::Int32,
        sizes::Ptr{Int64}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, N}(context, ptr, Dims{N}(sizes))
end

function Base.collect(input::CompasArray{Float32, N}) where {N}
    result = Array{Float32, N}(undef, reverse(input.sizes)...)
    @ccall LIBRARY.compas_read_array_float(
        input.context::Ptr{Cvoid},
        input::Ptr{Cvoid},
        result::Ptr{Float32},
        length(result)::Int64
    )::Cvoid
    return result
end

function Base.collect(input::CompasArray{ComplexF32, N}) where {N}
    result = Array{ComplexF32, N}(undef, reverse(input.sizes)...)
    @ccall LIBRARY.compas_read_array_complex(
        input.context::Ptr{Cvoid},
        input::Ptr{Cvoid},
        result::Ptr{ComplexF32},
        length(result)::Int64
    )::Cvoid
    return result
end

function assert_size(input::AbstractArray, expected::Dims)
    gotten = size(input)
    if gotten != expected
        throw(ArgumentError("Invalid argument dimensions $gotten, should be $expected"))
    end
end

function convert_array(input::CompasArray{T,N}, ty::Type{T}, dims::Integer...)::CompasArray{T,N} where {T,N}
    assert_size(input, Dims(dims))
    return input
end

function convert_array(input::Array{T,N}, ty::Type{T}, dims::Integer...)::CompasArray{T,N} where {T,N}
    assert_size(input, Dims(dims))
    context = get_context()
    return make_array(context, input)
end

function convert_array(input::AbstractArray, ty::Type{T}, dims::Integer...,)::CompasArray where {T}
    N = length(dims)
    return convert_array(convert(Array{T,N}, input), ty, dims...)
end

function convert_array_host(
    input::Array{T,N},
    ty::Type{T},
    dims::Dims{N},
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return input
end

function convert_array_host(
    input::AbstractArray,
    ty::Type{T},
    dims::Dims{N},
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return convert(Array{T,N}, input)
end

function convert_array_host(input::Number, ty::Type{T}, dims::Dims{N})::Array{T,N} where {T,N}
    return fill(convert(ty, input), dims)
end

abstract type Trajectory end

mutable struct CartesianTrajectory <: Trajectory
    ptr::Ptr{Cvoid}
    nreadouts::Int32
    samples_per_readout::Int32

    function CartesianTrajectory(
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::Number,
        k_start::AbstractVector,
        delta_k::Number
    )
        k_start = convert_array(k_start, ComplexF32, nreadouts)

        ptr = @ccall LIBRARY.compas_make_cartesian_trajectory(
            nreadouts::Int32,
            samples_per_readout::Int32,
            delta_t::Float32,
            k_start::Ptr{Cvoid},
            delta_k::Float32
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts, samples_per_readout)
        finalizer(unsafe_destroy_object!, obj)
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::CartesianTrajectory) = x.ptr

mutable struct SpiralTrajectory <: Trajectory
    ptr::Ptr{Cvoid}
    nreadouts::Int32
    samples_per_readout::Int32

    function SpiralTrajectory(
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::Number,
        k_start::AbstractVector,
        delta_k::AbstractVector
    )
        k_start = convert_array(k_start, ComplexF32, nreadouts)
        delta_k = convert_array(delta_k, ComplexF32, nreadouts)

        ptr = @ccall LIBRARY.compas_make_spiral_trajectory(
            nreadouts::Int32,
            samples_per_readout::Int32,
            delta_::Float32,
            k_start::Ptr{Cvoid},
            delta_k::Ptr{Cvoid},
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts, samples_per_readout)
        finalizer(unsafe_destroy_object!, obj)
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::SpiralTrajectory) = x.ptr

mutable struct TissueParameters
    ptr::Ptr{Cvoid}
    nvoxels::Int32

    function TissueParameters(
        nvoxels::Integer,
        T1::AbstractVector,
        T2::AbstractVector,
        B1::AbstractVector,
        B0::AbstractVector,
        rho_x::AbstractVector,
        rho_y::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        z::AbstractVector,
    )
        context = get_context()
        T1 = convert_array_host(T1, Float32, (nvoxels,))
        T2 = convert_array_host(T2, Float32, (nvoxels,))
        B1 = convert_array_host(B1,Float32, (nvoxels,))
        B0 = convert_array_host(B0, Float32, (nvoxels,))
        rho_x = convert_array_host(rho_x, Float32, (nvoxels,))
        rho_y = convert_array_host(rho_y, Float32, (nvoxels,))
        x = convert_array_host(x, Float32, (nvoxels,))
        y = convert_array_host(y, Float32, (nvoxels,))
        z = convert_array_host(z, Float32, (nvoxels,))

        ptr = @ccall LIBRARY.compas_make_tissue_parameters(
            context::Ptr{Cvoid},
            nvoxels::Int32,
            T1::Ptr{Float32},
            T2::Ptr{Float32},
            B1::Ptr{Float32},
            B0::Ptr{Float32},
            rho_x::Ptr{Float32},
            rho_y::Ptr{Float32},
            x::Ptr{Float32},
            y::Ptr{Float32},
            z::Ptr{Float32},
        )::Ptr{Cvoid}

        obj = new(ptr, nvoxels)
        finalizer(unsafe_destroy_object!, obj)
    end

    function TissueParameters(nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y)
        z = fill(0.0f0, nvoxels)
        return TissueParameters(nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y, z)
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::TissueParameters) = x.ptr

mutable struct FispSequence
    ptr::Ptr{Cvoid}
    nreadouts::Int32

    function FispSequence(
        RF_train::AbstractVector,
        slice_profiles::AbstractMatrix,
        TR::Number,
        TE::Number,
        max_state::Integer,
        TI::Number;
        undersampling_factor::Number=1,
        repetitions::Number=1,
    )
        RF_length = size(RF_train, 1)
        nreadouts = RF_length * undersampling_factor
        nslices = size(slice_profiles, 2)
        RF_train = convert_array(RF_train, ComplexF32, RF_length)
        slice_profiles = convert_array(slice_profiles, ComplexF32, RF_length, nslices)
        TW = 0.0

        ptr = @ccall LIBRARY.compas_make_fisp_sequence(
            RF_train::Ptr{Cvoid},
            slice_profiles::Ptr{Cvoid},
            TR::Float32,
            TE::Float32,
            TW::Float32,
            max_state::Int32,
            TI::Float32,
            undersampling_factor::Int32,
            repetitions::Int32,
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts)
        finalizer(unsafe_destroy_object!, obj)
    end
end

Base.unsafe_convert(T::Type{Ptr{Cvoid}}, x::FispSequence) = x.ptr

mutable struct pSSFPSequence
    ptr::Ptr{Cvoid}
    nreadouts::Int32

    function pSSFPSequence(
        RF_train::AbstractVector,
        TR::AbstractFloat,
        gamma_dt_RF::AbstractVector,
        dt::NTuple{3,<:AbstractFloat},
        gamma_dt_GRz::NTuple{3,<:AbstractFloat},
        z::AbstractVector,
    )
        nreadouts = size(RF_train, 1)
        nRF = size(gamma_dt_RF, 1)
        nslices = size(z, 1)

        RF_train = convert_array(RF_train, ComplexF32, nreadouts)
        gamma_dt_RF = convert_array(gamma_dt_RF, ComplexF32, nRF)
        z = convert_array(z, ComplexF32, nslices)

        ptr = @ccall LIBRARY.compas_make_pssfp_sequence(
            RF_train::Ptr{Cvoid},
            TR::Float32,
            gamma_dt_RF::Ptr{Cvoid},
            dt[1]::Float32,
            dt[2]::Float32,
            dt[3]::Float32,
            gamma_dt_GRz[1]::Float32,
            gamma_dt_GRz[2]::Float32,
            gamma_dt_GRz[3]::Float32,
            z::Ptr{Cvoid}
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts)
        finalizer(unsafe_destroy_object!, obj)
    end
end

Base.unsafe_convert(T::Type{Ptr{Cvoid}}, x::pSSFPSequence) = x.ptr

function simulate_magnetization(
    parameters::TissueParameters,
    sequence::FispSequence,
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts

    echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_fisp(
        context::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        sequence::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, echos_ptr, (nreadouts, nvoxels))
end

function simulate_magnetization(
    parameters::TissueParameters,
    sequence::pSSFPSequence,
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts

    echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_pssfp(
        context::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        sequence::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, echos_ptr, (nreadouts, nvoxels))
end

function simulate_magnetization_derivative(
    field::Integer,
    echos::AbstractMatrix{ComplexF32},
    parameters::TissueParameters,
    sequence::FispSequence,
    Δ::AbstractFloat
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts
    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)

    𝜕echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_derivative_fisp(
        context::Ptr{Cvoid},
        field::Int32,
        echos::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        Δ::Float32,
        sequence::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, 𝜕echos_ptr, (nreadouts, nvoxels))
end

function simulate_magnetization_derivative(
    field::Integer,
    echos::AbstractMatrix{ComplexF32},
    parameters::TissueParameters,
    sequence::pSSFPSequence,
    Δ::AbstractFloat
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts
    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)

    𝜕echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_derivative_pssfp(
        context::Ptr{Cvoid},
        field::Int32,
        echos::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        Δ::Float32,
        sequence::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, 𝜕echos_ptr, (nreadouts, nvoxels))
end

function simulate_magnetization_derivatives(
    echos::AbstractMatrix{ComplexF32},
    parameters::TissueParameters,
    sequence,
    Δ::AbstractFloat
)::@NamedTuple{
    T1::CompasArray{ComplexF32, 2},
    T2::CompasArray{ComplexF32, 2}
}
    return (
        T1 = simulate_magnetization_derivative(0, echos, parameters, sequence, Δ),
        T2 = simulate_magnetization_derivative(1, echos, parameters, sequence, Δ)
    )
end

function simulate_magnetization_derivatives(
    echos::AbstractMatrix{ComplexF32},
    parameters::TissueParameters,
    sequence,
)::@NamedTuple{
    T1::CompasArray{ComplexF32, 2},
    T2::CompasArray{ComplexF32, 2}
}
    return simulate_magnetization_derivatives(echos, parameters, sequence, 1e-4)
end


function magnetization_to_signal(
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::CartesianTrajectory,
    coils::AbstractMatrix,
)::CompasArray{ComplexF32, 3}
    context = get_context()
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)
    coils = convert_array(coils, ComplexF32, nvoxels, ncoils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_cartesian(
        context::Ptr{Cvoid},
        ncoils::Int32,
        echos::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        coils::Ptr{Cvoid},
        trajectory::Ptr{Cvoid},
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 3}(context, signal_ptr, (ncoils,  nreadouts, samples_per_readout))
end

function magnetization_to_signal(
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::SpiralTrajectory,
    coils::AbstractMatrix,
)::CompasArray{ComplexF32, 3}
    context = get_context()
    ncoils::Int64 = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)
    coils = convert_array(coils, ComplexF32, nvoxels, ncoils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_spiral(
        context::Ptr{Cvoid},
        ncoils::Int32,
        echos::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        coils::Ptr{Cvoid},
        trajectory::Ptr{Cvoid},
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 3}(context, signal_ptr, (ncoils,  nreadouts, samples_per_readout))
end

function compute_jacobian(
    echos::AbstractMatrix,
    𝜕echos::NamedTuple{(:T1, :T2)},
    parameters::TissueParameters,
    trajectory::Trajectory,
    coils::AbstractMatrix,
    v::AbstractMatrix
)::CompasArray{ComplexF32, 3}
    context = get_context()
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)
    𝜕echos_T1 = convert_array(𝜕echos.T1, ComplexF32, nvoxels, nreadouts)
    𝜕echos_T2 = convert_array(𝜕echos.T2, ComplexF32, nvoxels, nreadouts)
    coils = convert_array(coils, ComplexF32, nvoxels, ncoils)
    v = convert_array(v, ComplexF32, nvoxels, 4)

    Jv_ptr = @ccall LIBRARY.compas_compute_jacobian(
        context::Ptr{Cvoid},
        ncoils::Int32,
        echos::Ptr{Cvoid},
        𝜕echos_T1::Ptr{Cvoid},
        𝜕echos_T2::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        coils::Ptr{Cvoid},
        trajectory::Ptr{Cvoid},
        v::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 3}(context, Jv_ptr, (ncoils, nreadouts, samples_per_readout))
end

function compute_jacobian_hermitian(
    echos::AbstractMatrix,
    𝜕echos::NamedTuple{(:T1, :T2)},
    parameters::TissueParameters,
    trajectory::Trajectory,
    coils::AbstractMatrix,
    v::AbstractArray{<:Any,3}
)::CompasArray{ComplexF32, 2}
    context = get_context()
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)
    𝜕echos_T1 = convert_array(𝜕echos.T1, ComplexF32, nvoxels, nreadouts)
    𝜕echos_T2 = convert_array(𝜕echos.T2, ComplexF32, nvoxels, nreadouts)
    coils = convert_array(coils, ComplexF32, nvoxels, ncoils)
    v = convert_array(v, ComplexF32, samples_per_readout, nreadouts, ncoils)

    Jᴴv_ptr = @ccall LIBRARY.compas_compute_jacobian_hermitian(
        context::Ptr{Cvoid},
        ncoils::Int32,
        echos::Ptr{Cvoid},
        𝜕echos_T1::Ptr{Cvoid},
        𝜕echos_T2::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        trajectory::Ptr{Cvoid},
        coils::Ptr{Cvoid},
        v::Ptr{Cvoid}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, Jᴴv_ptr, (4, nvoxels))
end

function phase_encoding(
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::Trajectory
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout

    echos = convert_array(echos, ComplexF32, nvoxels, nreadouts)

    phe_echos_ptr = @ccall LIBRARY.phase_encoding(
        context::Ptr{Cvoid},
        echos::Ptr{Cvoid},
        parameters::Ptr{Cvoid},
        trajectory::Ptr{Cvoid},
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, phe_echos_ptr, (nreadouts, nvoxels))
end

function phase_encoding(
    inputs::NamedTuple,
    parameters::TissueParameters,
    trajectory::Trajectory
)::NamedTuple
    # Apply phase_encoding to each element in `inputs`
    return typeof(inputs)(
        map(v -> phase_encoding(v, parameters, trajectory), inputs)
    )
end

function compute_residual(
    lhs::AbstractArray{<:Any,3},
    rhs::AbstractArray{<:Any,3}
)::Tuple{Float32, CompasArray{ComplexF32, 3}}
    context = get_context()
    n, m, k = size(lhs)

    lhs = convert_array(lhs, ComplexF32, n, m, k)
    rhs = convert_array(rhs, ComplexF32, n, m, k)
    objective = [0.0f0]

    diff_ptr = @ccall LIBRARY.compas_compute_residual(
        context::Ptr{Cvoid},
        lhs::Ptr{Cvoid},
        rhs::Ptr{Cvoid},
        objective::Ptr{Cvoid}
    )::Ptr{Cvoid}

    diff = CompasArray{ComplexF32, 3}(context, diff_ptr, (k, m, n))

    return objective[1], diff
end

Base.pointer(c::Context) = c.ptr
Base.pointer(c::TissueParameters) = c.ptr
Base.pointer(c::CompasArray) = c.ptr

end
