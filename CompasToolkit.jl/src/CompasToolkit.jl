module CompasToolkit
include("Constants.jl")

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

function version()::String
    ptr = @ccall LIBRARY.compas_version()::Cstring
    return unsafe_string(ptr)
end

function unsafe_destroy_object!(obj)
    @ccall LIBRARY.compas_destroy(obj.ptr::Ptr{Cvoid})::Cvoid
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
Object representing an array of size `N` and type `T`.
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

Base.size(array::CompasArray) = reverse(array.sizes)
Base.getindex(array::CompasArray, i) = throw(ArgumentError("cannot index into a 'CompasArray'"))

function make_array(context::Context, input::Array{Float32, N})::CompasArray{Float32, N} where {N}
    sizes::Vector{Int64} = [reverse(size(input))...]

    ptr = @ccall LIBRARY.compas_make_array_float(
        context.ptr::Ptr{Cvoid},
        pointer(input)::Ptr{Float32},
        N::Int32,
        pointer(sizes)::Ptr{Int64}
    )::Ptr{Cvoid}

    return CompasArray{Float32, N}(context, ptr, Dims{N}(sizes))
end

function make_array(context::Context, input::Array{ComplexF32, N})::CompasArray{ComplexF32, N} where {N}
    sizes::Vector{Int64} = [reverse(size(input))...]

    ptr = @ccall LIBRARY.compas_make_array_complex(
        context.ptr::Ptr{Cvoid},
        pointer(input)::Ptr{ComplexF32},
        N::Int32,
        pointer(sizes)::Ptr{Int64}
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, N}(context, ptr, Dims{N}(sizes))
end

function Base.collect(input::CompasArray{Float32, N}) where {N}
    result = Array{Float32, N}(undef, reverse(input.sizes)...)
    @ccall LIBRARY.compas_read_array_float(
        pointer(input.context)::Ptr{Cvoid},
        input.ptr::Ptr{Cvoid},
        pointer(result)::Ptr{Float32},
        length(result)::Int64
    )::Cvoid
    return result
end

function Base.collect(input::CompasArray{ComplexF32, N}) where {N}
    result = Array{ComplexF32, N}(undef, reverse(input.sizes)...)
    @ccall LIBRARY.compas_read_array_complex(
        pointer(input.context)::Ptr{Cvoid},
        input.ptr::Ptr{Cvoid},
        pointer(result)::Ptr{ComplexF32},
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
    assert_size(input, dims)
    return input
end

function convert_array(input::Array{T,N}, ty::Type{T}, dims::Integer...)::CompasArray{T,N} where {T,N}
    assert_size(input, dims)
    context = get_context()
    return make_array(context, input)
end

function convert_array(input::AbstractArray, ty::Type{T}, dims::Integer...,)::CompasArray where {T}
    N = length(dims)
    return convert_array(convert(Array{T,N}, input), ty, dims...)
end

function convert_array_host(
    ty::Type{T},
    dims::Dims{N},
    input::Array{T,N},
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return input
end

function convert_array_host(
    ty::Type{T},
    dims::Dims{N},
    input::AbstractArray,
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return convert(Array{T,N}, input)
end

function convert_array_host(ty::Type{T}, dims::Dims{N}, input::Number)::Array{T,N} where {T,N}
    return fill(convert(ty, input), dims)
end

abstract type Trajectory end

mutable struct CartesianTrajectory <: Trajectory
    context::Context
    nreadouts::Int32
    samples_per_readout::Int32
    delta_t::Float32
    k_start::CompasArray{ComplexF32}
    delta_k::ComplexF32

    function CartesianTrajectory(
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::AbstractFloat,
        k_start::AbstractVector,
        delta_k::Number
    )
        return new(
            get_context(),
            nreadouts,
            samples_per_readout,
            delta_t,
            convert_array(k_start, ComplexF32, nreadouts),
            delta_k
        )
    end
end

mutable struct SpiralTrajectory <: Trajectory
    context::Context
    nreadouts::Int32
    samples_per_readout::Int32
    delta_t::Float32
    k_start::CompasArray{ComplexF32}
    delta_k::CompasArray{ComplexF32}

    function SpiralTrajectory(
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::AbstractFloat,
        k_start::AbstractVector,
        delta_k::AbstractVector
    )
        return new(
            get_context(),
            nreadouts,
            samples_per_readout,
            delta_t,
            convert_array(k_start, ComplexF32, nreadouts),
            convert_array(delta_k, ComplexF32, nreadouts)
        )
    end
end


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
        T1 = convert_array_host(Float32, (nvoxels,), T1)
        T2 = convert_array_host(Float32, (nvoxels,), T2)
        B1 = convert_array_host(Float32, (nvoxels,), B1)
        B0 = convert_array_host(Float32, (nvoxels,), B0)
        rho_x = convert_array_host(Float32, (nvoxels,), rho_x)
        rho_y = convert_array_host(Float32, (nvoxels,), rho_y)
        x = convert_array_host(Float32, (nvoxels,), x)
        y = convert_array_host(Float32, (nvoxels,), y)
        z = convert_array_host(Float32, (nvoxels,), z)

        ptr = @ccall LIBRARY.compas_make_tissue_parameters(
            pointer(context)::Ptr{Cvoid},
            nvoxels::Int32,
            pointer(T1)::Ptr{Float32},
            pointer(T2)::Ptr{Float32},
            pointer(B1)::Ptr{Float32},
            pointer(B0)::Ptr{Float32},
            pointer(rho_x)::Ptr{Float32},
            pointer(rho_y)::Ptr{Float32},
            pointer(x)::Ptr{Float32},
            pointer(y)::Ptr{Float32},
            pointer(z)::Ptr{Float32},
        )::Ptr{Cvoid}

        obj = new(ptr, nvoxels)
        finalizer(unsafe_destroy_object!, obj)
    end

    function TissueParameters(nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y)
        z = fill(0.0f0, nvoxels)
        return TissueParameters(nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y, z)
    end
end

mutable struct FispSequence
    context::Context
    nreadouts::Int32
    RF_train::CompasArray{ComplexF32, 1}
    slice_profiles::CompasArray{ComplexF32, 2}
    TR::Float32
    TE::Float32
    max_state::Int32
    TI::Float32

    function FispSequence(
        RF_train::AbstractVector,
        slice_profiles::AbstractMatrix,
        TR::Number,
        TE::Number,
        max_state::Integer,
        TI::Number,
    )
        nreadouts = size(RF_train, 1)
        nslices = size(slice_profiles, 2)

        return new(
            get_context(),
            nreadouts,
            convert_array(RF_train, ComplexF32, nreadouts),
            convert_array(slice_profiles, ComplexF32, nreadouts, nslices),
            TR,
            TE,
            max_state,
            TI)
    end
end

mutable struct pSSFPSequence
    context::Context
    RF_train::CompasArray{ComplexF32, 1}
    nreadouts::Int32
    TR::Float32
    nRF::Int32
    gamma_dt_RF::CompasArray{ComplexF32, 1}
    dt::NTuple{3, Float32}
    gamma_dt_GRz::NTuple{3, Float32}
    z::CompasArray{Float32, 1}

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

        return new(
            get_context(),
            convert_array(RF_train, ComplexF32, nreadouts),
            nreadouts,
            TR,
            nRF,
            convert_array(gamma_dt_RF, ComplexF32, nRF),
            dt,
            gamma_dt_GRz,
            convert_array(z, Float32, nslices)
        )
    end
end

function simulate_magnetization(
    parameters::TissueParameters,
    sequence::FispSequence,
)::CompasArray{ComplexF32, 2}
    context = get_context()
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts

    echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_fisp(
        pointer(context)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        sequence.RF_train.ptr::Ptr{Cvoid},
        sequence.slice_profiles.ptr::Ptr{Cvoid},
        sequence.TR::Float32,
        sequence.TE::Float32,
        sequence.max_state::Int32,
        sequence.TI::Float32
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
        pointer(context)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        sequence.RF_train.ptr::Ptr{Cvoid},
        sequence.TR::Float32,
        sequence.gamma_dt_RF.ptr::Ptr{Cvoid},
        sequence.dt[1]::Float32,
        sequence.dt[2]::Float32,
        sequence.dt[3]::Float32,
        sequence.gamma_dt_GRz[1]::Float32,
        sequence.gamma_dt_GRz[2]::Float32,
        sequence.gamma_dt_GRz[3]::Float32,
        sequence.z.ptr::Ptr{Cvoid}
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
        pointer(context)::Ptr{Cvoid},
        field::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        Δ::Float32,
        sequence.RF_train.ptr::Ptr{Cvoid},
        sequence.slice_profiles.ptr::Ptr{Cvoid},
        sequence.TR::Float32,
        sequence.TE::Float32,
        sequence.max_state::Int32,
        sequence.TI::Float32
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
        pointer(context)::Ptr{Cvoid},
        field::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        Δ::Float32,
        sequence.RF_train.ptr::Ptr{Cvoid},
        sequence.TR::Float32,
        sequence.gamma_dt_RF.ptr::Ptr{Cvoid},
        sequence.dt[1]::Float32,
        sequence.dt[2]::Float32,
        sequence.dt[3]::Float32,
        sequence.gamma_dt_GRz[1]::Float32,
        sequence.gamma_dt_GRz[2]::Float32,
        sequence.gamma_dt_GRz[3]::Float32,
        sequence.z.ptr::Ptr{Cvoid}
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
    coils = convert_array(coils, Float32, nvoxels, ncoils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_cartesian(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Cvoid},
        nreadouts::Int32,
        trajectory.samples_per_readout::Int32,
        trajectory.delta_t::Float32,
        trajectory.k_start.ptr::Ptr{Cvoid},
        trajectory.delta_k::ComplexF32,
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
    coils = convert_array(coils, Float32, nvoxels, ncoils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_spiral(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Float32},
        nreadouts::Int32,
        trajectory.samples_per_readout::Int32,
        trajectory.delta_t::Float32,
        trajectory.k_start.ptr::Ptr{Cvoid},
        trajectory.delta_k.ptr::Ptr{Cvoid},
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
    coils = convert_array(coils, Float32, nvoxels, ncoils)
    v = convert_array(v, ComplexF32, nvoxels, 4)

    Jv_ptr = @ccall LIBRARY.compas_compute_jacobian(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        pointer(𝜕echos_T1)::Ptr{Cvoid},
        pointer(𝜕echos_T2)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Cvoid},
        trajectory.nreadouts::Int32,
        trajectory.samples_per_readout::Int32,
        trajectory.delta_t::Float32,
        trajectory.k_start.ptr::Ptr{Cvoid},
        trajectory.delta_k::ComplexF32,
        pointer(v)::Ptr{Cvoid}
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
    coils = convert_array(coils, Float32, nvoxels, ncoils)
    v = convert_array(v, ComplexF32, samples_per_readout, nreadouts, ncoils)

    Jᴴv_ptr = @ccall LIBRARY.compas_compute_jacobian_hermitian(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        pointer(𝜕echos_T1)::Ptr{Cvoid},
        pointer(𝜕echos_T2)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Cvoid},
        trajectory.nreadouts::Int32,
        trajectory.samples_per_readout::Int32,
        trajectory.delta_t::Float32,
        pointer(trajectory.k_start)::Ptr{Cvoid},
        trajectory.delta_k::ComplexF32,
        pointer(v)::Ptr{Cvoid}
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
        pointer(context)::Ptr{Cvoid},
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        trajectory.nreadouts::Int32,
        trajectory.samples_per_readout::Int32,
        trajectory.delta_t::Float32,
        pointer(trajectory.k_start)::Ptr{Cvoid},
        trajectory.delta_k::ComplexF32
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
        pointer(context)::Ptr{Cvoid},
        lhs.ptr::Ptr{Cvoid},
        rhs.ptr::Ptr{Cvoid},
        pointer(objective)::Ptr{Float32}
    )::Ptr{Cvoid}

    diff = CompasArray{ComplexF32, 3}(context, diff_ptr, (k, m, n))

    return objective[1], diff
end

Base.pointer(c::Context) = c.ptr
Base.pointer(c::TissueParameters) = c.ptr
Base.pointer(c::CompasArray) = c.ptr

end
