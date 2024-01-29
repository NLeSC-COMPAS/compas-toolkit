using CompasToolkit
using ImagePhantoms
using LinearAlgebra

include("common.jl")



context = CompasToolkit.make_context()

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
nvoxels = N * N
T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
parameters_ref = map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y)
parameters = CompasToolkit.TissueParameters(context, nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

# Next, we assemble a Cartesian trajectory with linear phase encoding
trajectory_ref = generate_cartesian_trajectory(N);
trajectory = CompasToolkit.CartesianTrajectory(context,
    trajectory_ref.nreadouts, 
    trajectory_ref.nsamplesperreadout, 
    Float32(trajectory_ref.Δt), 
    ComplexF32.(trajectory_ref.k_start_readout), 
    ComplexF32.(trajectory_ref.Δk_adc[1]));

# We use four different receive coils
ncoils = 4
coil_sensitivities = generate_coils(N, ncoils)
coil_sensitivities_ref = map(SVector{ncoils}, eachrow(coil_sensitivities)) 

# isochromat model
pssfp_ref = generate_pssfp_sequence(N)
echos = generate_echos(N, pssfp_ref)
∂echos = generate_delta_echos(N, pssfp_ref)

v = rand(ComplexF32, nsamples(trajectory), ncoils)

Jᴴv = CompasToolkit.compute_jacobian_transposed(
    context,
    echos,
    ∂echos,
    parameters,
    trajectory,
    coil_sensitivities,
    v
)