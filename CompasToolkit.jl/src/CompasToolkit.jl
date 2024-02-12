module CompasToolkit
include("Constants.jl")

function __init__()
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

mutable struct Context
    ptr::Ptr{Cvoid}

    function Context(device::Integer)
        ptr = @ccall LIBRARY.compas_make_context(device::Int32)::Ptr{Cvoid}
        obj = new(ptr)
        destroy = (obj) -> @ccall LIBRARY.compas_destroy_context(ptr::Ptr{Cvoid})::Cvoid
        finalizer(destroy, obj)
    end
end

make_context(device::Integer)::Context = Context(device)
make_context() = make_context(0)

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

function assert_size(input::AbstractArray, expected::Dims{N}) where {N}
    gotten = size(input)
    if gotten != expected
        throw(ArgumentError("Invalid argument dimensions $gotten, should be $expected"))
    end
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

function convert_array(
    context::Context,
    ty::Type{T},
    dims::Dims{N},
    input::Array{T,N},
)::CompasArray{T,N} where {T,N}
    assert_size(input, dims)
    return make_array(context, input)
end

function unsafe_destroy_object!(obj)
    @ccall LIBRARY.compas_destroy(obj.ptr::Ptr{Cvoid})::Cvoid
end

abstract type Trajectory end

mutable struct CartesianTrajectory <: Trajectory
    context::Context
    nreadouts::Int32,
    samples_per_readout::Int32,
    delta_t::Float32,
    k_start::CompasArray{ComplexF32},
    delta_k::ComplexF32

    function CartesianTrajectory(
        context::Context
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::AbstractFloat,
        k_start::AbstractVector,
        delta_k::Number
    )
        k_start = convert_array(context, ComplexF32, (nreadouts,), k_start)

        return new(
            context,
            nreadouts,
            samples_per_readout,
            delta_t,
            k_start,
            delta_k
        )
    end
end

mutable struct SpiralTrajectory <: Trajectory
    context::Context
    nreadouts::Int32,
    samples_per_readout::Int32,
    delta_t::Float32,
    k_start::CompasArray{ComplexF32},
    delta_k::CompasArray{ComplexF32}

    function CartesianTrajectory(
        context::Context
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::AbstractFloat,
        k_start::AbstractVector,
        delta_k::AbstractVector
    )
        k_start = convert_array(context, ComplexF32, (nreadouts,), k_start)
        delta_k = convert_array(context, ComplexF32, (nreadouts,), delta_k)

        return new(
            context,
            nreadouts,
            samples_per_readout,
            delta_t,
            k_start,
            delta_k
        )
    end
end


mutable struct TissueParameters
    ptr::Ptr{Cvoid}
    nvoxels::Int32

    function TissueParameters(
        context::Context,
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

    function TissueParameters(context, nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y)
        z = fill(0.0f0, nvoxels)
        return TissueParameters(context, nvoxels, T1, T2, B1, B0, rho_x, rho_y, x, y, z)
    end
end

mutable struct FispSequence
    RF_train::CompasArray{ComplexF32, 1}
    slice_profiles::CompasArray{ComplexF32, 2}
    TR::Float32
    TE::Float32
    max_state::Int32
    TI::Float32

    function FispSequence(
        context::Context,
        RF_train::AbstractVector,
        slice_profiles::AbstractMatrix,
        TR::Number,
        TE::Number,
        max_state::Integer,
        TI::Number,
    )
        nreadouts = size(RF_train, 1)
        nslices = size(slice_profiles, 2)

        RF_train = convert_array(context, ComplexF32, (nreadouts,), RF_train)
        slice_profiles = convert_array(context, ComplexF32, (nreadouts, nslices), slice_profiles)

        return new(RF_train, slice_profiles, TR, TE, max_state, TI)
    end
end

mutable struct pSSFPSequence
    RF_train::CompasArray{ComplexF32, 1}
    TR::Float32
    nRF::Int32
    nTR::Int32
    gamma_dt_RF::CompasArray{ComplexF32, 1}
    dt::NTuple{3, Float32}
    gamma_dt_GRz::NTuple{3, Float32}
    z::CompasArray{Float32, 1}

    function pSSFPSequence(
        context::Context,
        RF_train::AbstractVector,
        TR::Float32,
        gamma_dt_RF::AbstractVector,
        dt::NTuple{3,<:AbstractFloat},
        gamma_dt_GRz::NTuple{3,<:AbstractFloat},
        z::AbstractVector,
    )
        nreadouts = size(RF_train, 1)
        nRF = size(gamma_dt_RF, 1)
        nslices = size(z, 1)

        RF_train = convert_array(context, ComplexF32, (nreadouts,), RF_train)
        gamma_dt_RF = convert_array(context, ComplexF32, (nRF,), gamma_dt_RF)
        z = convert_array(context, Float32, (nslices,), z)

        return new(RF_train, TR, nRF, nTR, gamma_dt_RF, dt, gamma_dt_GRz, z)
    end
end

function simulate_magnetization(
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    sequence::FispSequence,
)::CompasArray{ComplexF32, 2}
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts

    @ccall LIBRARY.compas_simulate_magnetization_fisp(
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
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    sequence::pSSFPSequence,
)::CompasArray{ComplexF32, 2}
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts

    echos_ptr = @ccall LIBRARY.compas_simulate_magnetization_pssfp(
        pointer(context)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        sequence.RF_train.ptr::Ptr{Cvoid},
        sequence.TR::Float32,
        sequence.gamma_dt_RF.ptr::Ptr{Cvoid},
        sequence.dt[0]::Float32,
        sequence.dt[1]::Float32,
        sequence.dt[2]::Float32,
        sequence.gamma_dt_GRz[0]::Float32,
        sequence.gamma_dt_GRz[1]::Float32,
        sequence.gamma_dt_GRz[2]::Float32
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 2}(context, echos_ptr, (nreadouts, nvoxels))
end

function magnetization_to_signal(
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::CartesianTrajectory,
    coils::AbstractMatrix,
)::CompasArray{ComplexF32, 3}
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(context, ComplexF32, (nvoxels, nreadouts), echos)
    coils = convert_array(context, Float32, (nvoxels, ncoils), coils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_cartesian(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        trajectory.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Float32},
        nreadouts::Int32,
        samples_per_Readout::Int32,
        trajectory.delta_t::Float32,
        trajectory.k_start.ptr::Ptr{Cvoid},
        trajectory.delta_k::ComplexF32,
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 3}(context, signal_ptr, (ncoils,  nreadouts, samples_per_readout))
end

function magnetization_to_signal(
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::SpiralTrajectory,
    coils::AbstractMatrix,
)::CompasArray{ComplexF32, 3}
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    echos = convert_array(context, ComplexF32, (nvoxels, nreadouts), echos)
    coils = convert_array(context, Float32, (nvoxels, ncoils), coils)

    signal_ptr = @ccall LIBRARY.compas_magnetization_to_signal_spiral(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(echos)::Ptr{Cvoid},
        parameters.ptr::Ptr{Cvoid},
        trajectory.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Float32},
        nreadouts::Int32,
        samples_per_Readout::Int32,
        trajectory.delta_t::Float32,
        trajectory.k_start.ptr::Ptr{Cvoid},
        trajectory.delta_k.ptr::Ptr{Cvoid},
    )::Ptr{Cvoid}

    return CompasArray{ComplexF32, 3}(context, signal_ptr, (ncoils,  nreadouts, samples_per_readout))
end

Base.pointer(c::Context) = c.ptr
Base.pointer(c::Trajectory) = c.ptr
Base.pointer(c::TissueParameters) = c.ptr
Base.pointer(c::FispSequence) = c.ptr
Base.pointer(c::pSSFPSequence) = c.ptr
Base.pointer(c::CompasArray) = c.ptr

end
