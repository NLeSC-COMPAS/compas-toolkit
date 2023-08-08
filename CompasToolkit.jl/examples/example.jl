using CompasToolkit
using ImagePhantoms
using Serialization
using LinearAlgebra

context = CompasToolkit.make_context(0)

# First we assemble a Shepp Logan phantom with homogeneous T₁ and T₂
# but non-constant proton density and B₀
N = 256
ρ = ComplexF32.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())') |> vec;
T₁ = fill(0.85, N, N) .|> Float32 |> vec;
T₂ = fill(0.05, N, N) .|> Float32 |> vec;
B₀ = repeat(1:N,1,N) .|> Float32 |> vec;
B₁ = repeat(1:N,1,N) .|> Float32 |> vec;

# We also set the spatial coordinates for the phantom
FOVˣ, FOVʸ = 25.6, 25.6;
X = [x for x ∈ LinRange(-FOVˣ/2, FOVˣ/2, N), y ∈ 1:N] .|> Float32 |> vec;
Y = [y for x ∈ 1:N, y ∈ LinRange(-FOVʸ/2, FOVʸ/2, N)] .|> Float32 |> vec;
nvoxels = N*N

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
parameters = CompasToolkit.make_tissue_parameters(context, nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)


# Next, we assemble a Cartesian trajectory with linear phase encoding
nr = N # nr of readouts
ns = N # nr of samples per readout
Δt_adc = 10^-5 # time between sample points
py = -(N÷2):1:(N÷2)-1 # phase encoding indices
Δkˣ = 2π / FOVˣ; # k-space step in x direction for Nyquist sampling
Δkʸ = 2π / FOVʸ; # k-space step in y direction for Nyquist sampling
k0 = [(-ns/2 * Δkˣ) + im * (py[mod1(r,N)] * Δkʸ) for r in 1:nr]; # starting points in k-space per readout
Δk = [Δkˣ + 0.0im for r in 1:nr]; # k-space steps per sample point for each readout

trajectory = CompasToolkit.make_cartesian_trajectory(context, nr, ns, Float32(Δt_adc), ComplexF32.(k0), ComplexF32.(Δk));


# We use two different receive coils
ncoils = 4
coil₁ = Float32.(repeat(LinRange(0.5,1.0,N),1,N));
coil₂ = coil₁';
coil₃ = coil₂;
coil₄ = coil₂;

coil_sensitivities = hcat(coil₁ |> vec, coil₂ |> vec, coil₃ |> vec, coil₄ |> vec)
signal = zeros(ComplexF32, ns, nr, ncoils)

echos = zeros(ComplexF32, N*N, nr)
transpose!(echos, deserialize("echos.bin"))

CompasToolkit.simulate_signal(
    context,
    signal,
    echos,
    parameters,
    trajectory,
    coil_sensitivities)


signal_ref = reshape(deserialize("signal.bin"), ncoils, nr, ns)

for c in 1:ncoils
    println("match coil ", c, ": ", signal[:,:,c] ≈ signal_ref[c,:,:])
end
