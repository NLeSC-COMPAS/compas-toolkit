using BlochSimulators
using CompasToolkit
using ImagePhantoms
using ComputationalResources
using LinearAlgebra
using StaticArrays

include("common.jl")

context = CompasToolkit.make_context(0)

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
parameters_ref = map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y)
parameters = CompasToolkit.TissueParameters(context, nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

# Next, we assemble a balanced sequence with constant flip angle of 60 degrees,
pssfp_ref = generate_pssfp_sequence(N)

# isochromat model
echos = generate_echos(N, pssfp_ref)

# Next, we assemble a Cartesian trajectory with linear phase encoding
trajectory_ref = generate_cartesian_trajectory(N);
trajectory = CompasToolkit.CartesianTrajectory(context, 
    trajectory_ref.nr, 
    trajectory_ref.ns, 
    Float32(trajectory_ref.Δt_adc), 
    ComplexF32.(trajectory_ref.k0), 
    ComplexF32.(trajectory_ref.Δk[1]));

# We use two different receive coils
ncoils = 4
coil_sensitivities = generate_coils(N)
coil_sensitivities_ref = map(SVector{ncoils}, eachrow(coil_sensitivities)) 

trajectory_ref  = gpu(f32(trajectory_ref))
coil_sensitivities_ref  = gpu(f32(coil_sensitivities_ref))
signal_ref = simulate(CUDALibs(), pssfp_ref, parameters_ref, trajectory_ref, coil_sensitivities_ref)
signal_ref = reshape(collect(signal_ref), ns, nr)

signal = zeros(ComplexF32, ns, nr, ncoils)
CompasToolkit.magnetization_to_signal(
    context,
    signal,
    echos,
    parameters,
    trajectory,
    coil_sensitivities)


for c in 1:ncoils
    expected = map(x -> x[c], signal_ref)
    answer = signal[:,:,c]

    println("fraction equal: ", sum(isapprox.(answer, expected, rtol=0.05)) / length(answer))

    err = abs.(answer - expected)
    println("maximum abs error: ", maximum(err))
    println("maximum rel error: ", maximum(err ./ abs.(expected)))

    idx = argmax(err ./ abs.(expected))
    println("maximum rel error index: ", idx, " ", expected[idx], " != ", answer[idx])
end