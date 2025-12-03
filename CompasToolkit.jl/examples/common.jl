using BlochSimulators
using StaticArrays
using Statistics
using ComputationalResources
using ImagePhantoms
using LinearAlgebra
using Random
using Printf

Random.seed!(1337)

struct ∂mˣʸ∂T₁T₂{T} <: FieldVector{2, T}
    ∂T₁::T
    ∂T₂::T
end

struct ∂mˣʸ∂T₁T₂ρˣρʸ{T} <: FieldVector{4, T}
    ∂T₁::T
    ∂T₂::T
    ∂ρˣ::T
    ∂ρʸ::T
end

StaticArrays.similar_type(::Type{∂mˣʸ∂T₁T₂{T}}, ::Type{T}, s::Size{(2,)}) where T = ∂mˣʸ∂T₁T₂{T}
StaticArrays.similar_type(::Type{∂mˣʸ∂T₁T₂ρˣρʸ{T}}, ::Type{T}, s::Size{(4,)}) where T = ∂mˣʸ∂T₁T₂ρˣρʸ{T}

function generate_parameters(N)
    # First we assemble a Shepp Logan phantom with homogeneous T₁ and T₂
    # but non-constant proton density and B₀
    ρ = ComplexF32.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())') |> vec;
    T₁ = fill(0.85f0, N, N) |> vec;
    T₂ = fill(0.05f0, N, N) |> vec;
    B₀ = zeros(Float32, N, N) |> vec; #repeat(1:N,1,N) .|> Float32 |> vec;
    B₁ = ones(Float32, N, N) |> vec;

    FOVˣ, FOVʸ = 25.6, 25.6;
    X = [x for x ∈ LinRange(-FOVˣ/2, FOVˣ/2, N), y ∈ 1:N] .|> Float32 |> vec;
    Y = [y for x ∈ 1:N, y ∈ LinRange(-FOVʸ/2, FOVʸ/2, N)] .|> Float32 |> vec;

    return (T₁, T₂, B₁, B₀, ρ, X, Y)
end

function generate_pssfp_sequence(N)
    nTR = N
    RF_train = complex.(fill(40.0, nTR)) # constant flip angle train
    RF_train[2:2:end] .*= -1 # 0-π phase cycling
    nRF = 25 # nr of RF discretization points
    durRF = 0.001 # duration of RF excitation
    TR = 0.010 # repetition time
    TI = 10.0 # long inversion delay -> no inversion
    gaussian = [exp(-(i-(nRF/2))^2 * inv(nRF)) for i ∈ 1:nRF] # RF excitation waveform
    γΔtRF = (π/180) * normalize(gaussian, 1) |> SVector{nRF} # normalize to flip angle of 1 degree
    Δt = (ex=durRF/nRF, inv = TI, pr = (TR - durRF)/2); # time intervals during TR
    γΔtGRz = (ex=0.002/nRF, inv = 0.00, pr = -0.01); # slice select gradient strengths during TR
    nz = 35 # nr of spins in z direction
    z = SVector{nz}(LinRange(-1,1,nz)) # z locations

    return f32(pSSFP2D(RF_train, TR, γΔtRF, Δt, γΔtGRz, z))
end

function generate_cartesian_trajectory(N)
    FOVˣ, FOVʸ = 25.6, 25.6;
    FOVˣ, FOVʸ = 30.0, 28.0;

    nr = N # nr of readouts
    ns = N # nr of samples per readout
    Δt_adc = 10^-5 # time between sample points
    py = -(N÷2):1:(N÷2)-1 # phase encoding indices
    Δkˣ = 2π / FOVˣ; # k-space step in x direction for Nyquist sampling
    Δkʸ = 2π / FOVʸ; # k-space step in y direction for Nyquist sampling
    k0 = [(-ns/2 * Δkˣ) + im * (py[mod1(r,N)] * Δkʸ) for r in 1:nr]; # starting points in k-space per readout
    Δk = [Δkˣ + 0.0im for r in 1:nr]; # k-space steps per sample point for each readout
    
    return f32(CartesianTrajectory2D(nr, ns, Δt_adc, k0, Δkˣ, py, 2))
end

function generate_coils(N, ncoils)
    coil₁ = complex.(repeat(LinRange(0.5,1.0,N),1,N));
    coil₂ = coil₁';
    coil₃ = coil₁;
    coil₄ = coil₂;
    
    coils = hcat(coil₁ |> vec, coil₂ |> vec, coil₃ |> vec, coil₄ |> vec) .|> ComplexF32
    return coils[:, 1:ncoils]
end

function generate_echos(N, sequence)
    sequence = gpu(f32(sequence))
    
    # isochromat model
    T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
    parameters_ref = gpu(f32(map(T₁T₂B₀ρˣρʸ, T₁, T₂, B₀, real.(ρ), imag.(ρ))))
    
    echos_ref = simulate_magnetization(CUDALibs(), sequence, parameters_ref)
    return collect(transpose(echos_ref))
end

function generate_delta_echos(N, sequence)
    Δ = 0.001f0
    sequence = gpu(f32(sequence))

    # isochromat model
    T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)

    parameters_ref = gpu(f32(map(T₁T₂B₀ρˣρʸ, T₁, T₂, B₀, real.(ρ), imag.(ρ))))
    echos = simulate_magnetization(CUDALibs(), sequence, parameters_ref)

    parameters_ref = gpu(f32(map(T₁T₂B₀ρˣρʸ, T₁ .+ Δ, T₂, B₀, real.(ρ), imag.(ρ))))
    𝜕echos_dT1 = simulate_magnetization(CUDALibs(), sequence, parameters_ref) - echos

    parameters_ref = gpu(f32(map(T₁T₂B₀ρˣρʸ, T₁, T₂ .+ Δ, B₀, real.(ρ), imag.(ρ))))
    𝜕echos_dT2 = simulate_magnetization(CUDALibs(), sequence, parameters_ref) - echos

    return (
        T1=collect(transpose(𝜕echos_dT1))  ./ Δ,
        T2=collect(transpose(𝜕echos_dT2))  ./ Δ
    )
end

function print_equals_check(expected, answer; atol = 1e-9)
    answer = collect(answer)

    println("=== Statistics ===")
    # Shape and length are the same, so print once
    @printf("  Shape: %s\n", string(size(expected)))
    @printf("  Min:   %s vs %s\n", expected[argmin(real.(expected))], answer[argmin(real.(answer))])
    @printf("  Max:   %s vs %s\n", expected[argmax(real.(expected))], answer[argmax(real.(answer))])
    @printf("  Mean:  %s vs %s\n", mean(expected), mean(answer))
    @printf("  Std.:  %s vs %s\n", std(expected), std(answer))
    @printf("  Zeros: %s vs %s\n", mean(expected .== 0), mean(answer .== 0))
    println()

    println("=== Fraction of equal values ===")
    for rtol in [0.001, 0.005, 0.01, 0.05, 0.1]
        is_equal = isapprox.(answer, expected, atol=atol, rtol=rtol)
        println("  atol=$(atol), rtol=$(rtol): ", sum(is_equal) / length(answer))
    end

    err = abs.(answer .- expected)
    idx_abs = argmax(err)
    max_abs_err = err[idx_abs]

    println("\n=== Maximum absolute error ===")
    println("  Index:          $(Tuple(idx_abs))")
    println("  Answer:         $(answer[idx_abs])")
    println("  Expected:       $(expected[idx_abs])")
    println("  Absolute Error: $(max_abs_err)")
    println()

    rel_err = err ./ max.(abs.(expected), atol)
    idx_rel = argmax(rel_err)
    max_rel_err = rel_err[idx_rel]

    println("\n=== Maximum relative error ===")
    println("  Index:          $(Tuple(idx_rel))")
    println("  Answer:         $(answer[idx_rel])")
    println("  Expected:       $(expected[idx_rel])")
    println("  Relative Error: $max_rel_err")
    println()
    println()
end