using CompasToolkit
using StructArrays


include("common.jl")

context = CompasToolkit.init_context(0)

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
nvoxels = N * N
T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
parameters_ref = gpu(StructArray(f32(map(T₁T₂B₀ρˣρʸ, T₁, T₂, B₀, real.(ρ), imag.(ρ)))))
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

# We use four different receive coils
ncoils = 4
coil_sensitivities = generate_coils(N, ncoils)

echos_ref = gpu(transpose(echos))
trajectory_ref  = gpu(f32(trajectory_ref))
coil_sensitivities_ref  = gpu(f32(coil_sensitivities))
coordinates_ref = gpu(StructArray(Coordinates(x, y, zero(x)) for (x, y) in zip(X, Y)))

signal_ref = magnetization_to_signal(CUDALibs(), echos_ref, gpu(parameters_ref), trajectory_ref, coordinates_ref, coil_sensitivities_ref)
signal_ref = collect(signal_ref)
signal_ref = reshape(signal_ref, ns, nr, ncoils)

signal = CompasToolkit.magnetization_to_signal(
    echos,
    parameters,
    trajectory,
    coil_sensitivities)
signal = collect(signal)

print_equals_check(signal_ref, signal)
