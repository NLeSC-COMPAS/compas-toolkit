using BlochSimulators
using CompasToolkit
using ImagePhantoms
using ComputationalResources
using LinearAlgebra
using StaticArrays

context = CompasToolkit.make_context(0)

# First we assemble a Shepp Logan phantom with homogeneous T₁ and T₂
# but non-constant proton density and B₀
N = 256
ρ = ComplexF32.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())') |> vec;
T₁ = fill(0.85f0, N, N) |> vec;
T₂ = fill(0.05f0, N, N) |> vec;
B₀ = repeat(1:N,1,N) .|> Float32 |> vec;
B₁ = ones(Float32, N, N) |> vec;

# We also set the spatial coordinates for the phantom
FOVˣ, FOVʸ = 25.6, 25.6;
X = [x for x ∈ LinRange(-FOVˣ/2, FOVˣ/2, N), y ∈ 1:N] .|> Float32 |> vec;
Y = [y for x ∈ 1:N, y ∈ LinRange(-FOVʸ/2, FOVʸ/2, N)] .|> Float32 |> vec;
nvoxels = N*N

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
parameters_ref = map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y)
parameters = CompasToolkit.TissueParameters(context, nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

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
Δt = (ex=durRF/nRF, inv = TI, pr = (TR - durRF)/2); # time intervals during TR
γΔtGRz = (ex=0.002/nRF, inv = 0.00, pr = -0.01); # slice select gradient strengths during TR
nz = 35 # nr of spins in z direction
z = SVector{nz}(LinRange(-1,1,nz)) # z locations

pssfp_ref = pSSFP(RF_train, TR, γΔtRF, Δt, γΔtGRz, z)

# isochromat model
pssfp_ref = gpu(f32(pssfp_ref))
parameters_ref = gpu(f32(parameters_ref))
echos_ref = simulate(CUDALibs(), pssfp_ref, parameters_ref);
echos = collect(transpose(echos_ref))

# Next, we assemble a Cartesian trajectory with linear phase encoding
nr = N # nr of readouts
ns = N # nr of samples per readout
Δt_adc = 10^-5 # time between sample points
py = -(N÷2):1:(N÷2)-1 # phase encoding indices
Δkˣ = 2π / FOVˣ; # k-space step in x direction for Nyquist sampling
Δkʸ = 2π / FOVʸ; # k-space step in y direction for Nyquist sampling
k0 = [(-ns/2 * Δkˣ) + im * (py[mod1(r,N)] * Δkʸ) for r in 1:nr]; # starting points in k-space per readout
Δk = [Δkˣ + 0.0im for r in 1:nr]; # k-space steps per sample point for each readout

trajectory_ref = CartesianTrajectory(nr,ns,Δt_adc,k0,Δk,py);
trajectory = CompasToolkit.CartesianTrajectory(context, nr, ns, Float32(Δt_adc), ComplexF32.(k0), ComplexF32.(Δk[1]));

# We use two different receive coils
ncoils = 4
coil₁ = complex.(repeat(LinRange(0.5,1.0,N),1,N));
coil₂ = coil₁';
coil₃ = coil₂;
coil₄ = coil₂;

coil_sensitivities = hcat(coil₁ |> vec, coil₂ |> vec, coil₃ |> vec, coil₄ |> vec) .|> Float32
coil_sensitivities_ref = map(SVector, coil₁, coil₂, coil₃, coil₄)

trajectory_ref  = gpu(f32(trajectory_ref))
coil_sensitivities_ref  = gpu(f32(coil_sensitivities_ref))
signal_ref = simulate(CUDALibs(), pssfp_ref, parameters_ref, trajectory_ref, coil_sensitivities_ref)

signal_ref = collect(signal_ref)
signal_ref = reshape(signal_ref, ns, nr)

signal = zeros(ComplexF32, ns, nr, ncoils)
CompasToolkit.simulate_signal(
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