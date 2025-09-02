using CompasToolkit
using StructArrays


include("common.jl")


# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
nvoxels = N * N

# We use four different receive coils
coil₁ = complex.(repeat(LinRange(0.5,1.0,N),1,N));
coil₂ = coil₁'
coil_sensitivities = hcat(coil₁ |> vec, coil₂ |> vec) .|> ComplexF32

ρ = ComplexF32.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())') |> vec;
T₁ = fill(0.85f0, N, N) |> vec;
T₂ = fill(0.05f0, N, N) |> vec;
B₀ = zeros(Float32, N, N) |> vec;
B₁ = ones(Float32, N, N) |> vec;
FOVˣ, FOVʸ = 30.0, 28.0
X = [x for x ∈ LinRange(-FOVˣ/2, FOVˣ/2, N), y ∈ 1:N] .|> Float32 |> vec;
Y = [y for x ∈ 1:N, y ∈ LinRange(-FOVʸ/2, FOVʸ/2, N)] .|> Float32 |> vec;

# Next, we assemble a balanced sequence with constant flip angle of 60 degrees,
nTR = N
RF_train = complex.(fill(40.0, nTR)) # constant flip angle train
RF_train[2:2:end] .*= -1 # 0-π phase cycling
nRF = 25 # nr of RF discretization points
durRF = 0.001 # duration of RF excitation
TR = 0.010 # repetition time
TI = 10.0 # long inversion delay -> no inversion
gaussian = [exp(-(i-(nRF/2))^2 * inv(nRF)) for i ∈ 1:nRF] # RF excitation waveform
γΔtRF = (π/180) * normalize(gaussian, 1) |> SVector{nRF} # normalize to flip angle of 1 degree
Δt = (durRF/nRF, TI, (TR - durRF)/2); # time intervals during TR
γΔtGRz = (0.002/nRF, 0.00, -0.01); # slice select gradient strengths during TR
nz = 35 # nr of spins in z direction
z = SVector{nz}(LinRange(-1,1,nz)) # z locations

# isochromat model
sliceprofiles = ones(nTR,1) .|> complex;
TR, TE, TI = 0.010, 0.006, 0.025
max_state = 32

# Next, we assemble a Cartesian trajectory with linear phase encoding
FOVˣ, FOVʸ = 30.0, 28.0
nr, ns = N, N # nr of readouts,  nr of samples per readout
Δt_adc = 10^-5 # time between sample points
py = -(N÷2):1:(N÷2)-1 # phase encoding indices
Δkˣ = 2π / FOVˣ; # k-space step in x direction for Nyquist sampling
Δkʸ = 2π / FOVʸ; # k-space step in y direction for Nyquist sampling
k0 = [(-ns/2 * Δkˣ) + im * (py[mod1(r,N)] * Δkʸ) for r in 1:nr]; # starting points in k-space per readout


# Initialize Compas
CompasToolkit.init_context()

# Pass in our defined parameters and sequence
parameters = CompasToolkit.TissueParameters(nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)
pssfp = CompasToolkit.pSSFPSequence(RF_train, TR, γΔtRF, Δt, γΔtGRz, z)
sequence = CompasToolkit.FispSequence(RF_train, sliceprofiles, TR, TE, max_state, TI)
trajectory = CompasToolkit.CartesianTrajectory( nr, ns, Δt_adc, k0, Δkˣ)

# Simulate the MRI signal!
echos = CompasToolkit.simulate_magnetization(parameters, sequence)
signal = CompasToolkit.magnetization_to_signal(echos, parameters, trajectory, coil_sensitivities)

