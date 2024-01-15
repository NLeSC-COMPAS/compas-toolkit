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

function assert_size(input::AbstractArray, expected::Dims{N}) where {N}
    gotten = size(input)
    if gotten != expected
        throw(ArgumentError("Invalid argument dimensions $gotten, should be $expected"))
    end
end

function convert_array(
    ty::Type{T},
    dims::Dims{N},
    input::Array{T,N},
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return input
end

function convert_array(
    ty::Type{T},
    dims::Dims{N},
    input::AbstractArray,
)::Array{T,N} where {T,N}
    assert_size(input, dims)
    return convert(Array{T,N}, input)
end

function convert_array(ty::Type{T}, dims::Dims{N}, input::Number)::Array{T,N} where {T,N}
    return fill(convert(ty, input), dims)
end

function unsafe_destroy_object!(obj)
    @ccall LIBRARY.compas_destroy(obj.ptr::Ptr{Cvoid})::Cvoid
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

abstract type Trajectory end

mutable struct CartesianTrajectory <: Trajectory
    ptr::Ptr{Cvoid}
    nreadouts::Int32
    samples_per_readout::Int32

    function CartesianTrajectory(
        context::Context,
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::Number,
        k_start::AbstractVector,
        delta_k::Number,
    )
        k_start = convert_array(ComplexF32, (nreadouts,), k_start)

        ptr = @ccall LIBRARY.compas_make_cartesian_trajectory(
            pointer(context)::Ptr{Cvoid},
            nreadouts::Int32,
            samples_per_readout::Int32,
            delta_t::Float32,
            pointer(k_start)::Ptr{ComplexF32},
            delta_k::ComplexF32,
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts, samples_per_readout)
        finalizer(unsafe_destroy_object!, obj)
    end
end

mutable struct SpiralTrajectory <: Trajectory
    ptr::Ptr{Cvoid}
    nreadouts::Int32
    samples_per_readout::Int32

    function SpiralTrajectory(
        context::Context,
        nreadouts::Integer,
        samples_per_readout::Integer,
        delta_t::Number,
        k_start::AbstractVector,
        delta_k::AbstractVector,
    )
        k_start = convert_array(ComplexF32, (nreadouts,), k_start)
        delta_k = convert_array(ComplexF32, (nreadouts,), delta_k)

        ptr = @ccall LIBRARY.compas_make_spiral_trajectory(
            pointer(context)::Ptr{Cvoid},
            nreadouts::Int32,
            samples_per_readout::Int32,
            delta_t::Float32,
            pointer(k_start)::Ptr{ComplexF32},
            pointer(delta_k)::Ptr{ComplexF32},
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts, samples_per_readout)
        finalizer(unsafe_destroy_object!, obj)
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
        T1 = convert_array(Float32, (nvoxels,), T1)
        T2 = convert_array(Float32, (nvoxels,), T2)
        B1 = convert_array(Float32, (nvoxels,), B1)
        B0 = convert_array(Float32, (nvoxels,), B0)
        rho_x = convert_array(Float32, (nvoxels,), rho_x)
        rho_y = convert_array(Float32, (nvoxels,), rho_y)
        x = convert_array(Float32, (nvoxels,), x)
        y = convert_array(Float32, (nvoxels,), y)
        z = convert_array(Float32, (nvoxels,), z)


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
    ptr::Ptr{Cvoid}
    nreadouts::Int32

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

        RF_train = convert_array(ComplexF32, (nreadouts,), RF_train)
        slice_profiles = convert_array(ComplexF32, (nreadouts, nslices), slice_profiles)

        ptr = @ccall LIBRARY.compas_make_fisp_sequence(
            pointer(context)::Ptr{Cvoid},
            nreadouts::Int32,
            nslices::Int32,
            pointer(RF_train)::Ptr{ComplexF32},
            pointer(slice_profiles)::Ptr{ComplexF32},
            TR::Float32,
            TE::Float32,
            max_state::Int32,
            TI::Float32,
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts)
        finalizer(unsafe_destroy_object!, obj)
    end
end

mutable struct pSSFPSequence
    ptr::Ptr{Cvoid}
    nreadouts::Int32

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

        RF_train = convert_array(ComplexF32, (nreadouts,), RF_train)
        gamma_dt_RF = convert_array(ComplexF32, (nRF,), gamma_dt_RF)
        z = convert_array(ComplexF32, (nslices,), z)

        ptr = @ccall LIBRARY.compas_make_pssfp_sequence(
            pointer(context)::Ptr{Cvoid},
            nRF::Int32,
            nreadouts::Int32,
            nslices::Int32,
            pointer(RF_train)::Ptr{ComplexF32},
            TR::Float32,
            pointer(gamma_dt_RF)::Ptr{ComplexF32},
            dt[1]::Float32,
            dt[2]::Float32,
            dt[3]::Float32,
            gamma_dt_GRz[1]::Float32,
            gamma_dt_GRz[2]::Float32,
            gamma_dt_GRz[3]::Float32,
            pointer(z)::Ptr{Float32},
        )::Ptr{Cvoid}

        obj = new(ptr, nreadouts)
        finalizer(unsafe_destroy_object!, obj)
    end
end

function simulate_magnetization(
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    sequence::FispSequence,
)
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts
    echos = convert_array(ComplexF32, (nvoxels, nreadouts), echos)

    @ccall LIBRARY.compas_simulate_magnetization_fisp(
        pointer(context)::Ptr{Cvoid},
        pointer(echos)::Ptr{ComplexF32},
        parameters.ptr::Ptr{Cvoid},
        sequence.ptr::Ptr{Cvoid},
    )::Cvoid

    return echos
end

function simulate_magnetization(
    context::Context,
    echos::AbstractMatrix,
    parameters::TissueParameters,
    sequence::pSSFPSequence,
)
    nvoxels::Int64 = parameters.nvoxels
    nreadouts::Int64 = sequence.nreadouts
    echos = convert_array(ComplexF32, (nvoxels, nreadouts), echos)

    @ccall LIBRARY.compas_simulate_magnetization_pssfp(
        pointer(context)::Ptr{Cvoid},
        pointer(echos)::Ptr{ComplexF32},
        parameters.ptr::Ptr{Cvoid},
        sequence.ptr::Ptr{Cvoid},
    )::Cvoid

    return echos
end

function magnetization_to_signal(
    context::Context,
    signal::AbstractArray{<:Any,3},
    echos::AbstractMatrix,
    parameters::TissueParameters,
    trajectory::Trajectory,
    coils::AbstractMatrix,
)
    ncoils = size(coils, 2)
    nreadouts::Int64 = trajectory.nreadouts
    samples_per_readout::Int64 = trajectory.samples_per_readout
    nvoxels::Int64 = parameters.nvoxels

    signal = convert_array(ComplexF32, (samples_per_readout, nreadouts, ncoils), signal)
    echos = convert_array(ComplexF32, (nvoxels, nreadouts), echos)
    coils = convert_array(Float32, (nvoxels, ncoils), coils)

    @ccall LIBRARY.compas_magnetization_to_signal(
        pointer(context)::Ptr{Cvoid},
        ncoils::Int32,
        pointer(signal)::Ptr{ComplexF32},
        pointer(echos)::Ptr{ComplexF32},
        parameters.ptr::Ptr{Cvoid},
        trajectory.ptr::Ptr{Cvoid},
        pointer(coils)::Ptr{Float32},
    )::Cvoid

    return signal
end

Base.pointer(c::Context) = c.ptr
Base.pointer(c::Trajectory) = c.ptr
Base.pointer(c::TissueParameters) = c.ptr
Base.pointer(c::FispSequence) = c.ptr
Base.pointer(c::pSSFPSequence) = c.ptr

end
