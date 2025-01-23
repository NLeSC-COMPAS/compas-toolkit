using BlochSimulators
using CompasToolkit
using ImagePhantoms
using ComputationalResources
using LinearAlgebra
using StaticArrays

include("common.jl")

context = CompasToolkit.init_context(0)

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
nvoxels = N * N
T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
parameters_ref = gpu(f32(map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y)))
parameters = CompasToolkit.TissueParameters(nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

# Next, we assemble a balanced sequence with constant flip angle of 60 degrees,
pssfp_ref = generate_pssfp_sequence(N)

# isochromat model
echos = generate_echos(N, pssfp_ref)

# Next, we assemble a Cartesian trajectory with linear phase encoding
trajectory_ref = generate_cartesian_trajectory(N);
trajectory = CompasToolkit.CartesianTrajectory(
    trajectory_ref.nreadouts,
    trajectory_ref.nsamplesperreadout,
    trajectory_ref.Δt,
    trajectory_ref.k_start_readout,
    trajectory_ref.Δk_adc[1]);

nr = trajectory_ref.nreadouts
ns = trajectory_ref.nsamplesperreadout

# We use two different receive coils
ncoils = 4
coil_sensitivities = generate_coils(N, ncoils)
coil_sensitivities_ref = map(SVector{ncoils}, eachrow(coil_sensitivities)) 

trajectory_ref  = gpu(f32(trajectory_ref))
coil_sensitivities_ref  = gpu(f32(coil_sensitivities_ref))
signal_ref = simulate_signal(CUDALibs(), gpu(pssfp_ref), gpu(parameters_ref), trajectory_ref, coil_sensitivities_ref)
signal_ref = reshape(collect(signal_ref), ns, nr)

echos = gpu(transpose(echos))
phase_encoding!(echos, trajectory_ref, parameters_ref)
echos = collect(transpose(echos))
signal = CompasToolkit.magnetization_to_signal(
    echos,
    parameters,
    trajectory,
    coil_sensitivities)
signal = collect(signal)

for c in 1:ncoils
    expected = map(x -> x[c], signal_ref)
    answer = signal[:,:,c]
    print_equals_check(expected, answer)
end
