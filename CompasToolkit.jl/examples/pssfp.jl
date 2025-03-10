using CompasToolkit
using ImagePhantoms

include("common.jl")

context = CompasToolkit.init_context(0)

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
parameters_ref = map(T₁T₂B₀ρˣρʸ, T₁, T₂, B₀, real.(ρ), imag.(ρ))
parameters = CompasToolkit.TissueParameters(nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

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

pssfp_ref = pSSFP2D(RF_train, TR, γΔtRF, Δt, γΔtGRz, z)


Δt = (Δt.ex, Δt.inv, Δt.pr) # time intervals during TR
γΔtGRz = (γΔtGRz.ex, γΔtGRz.inv, γΔtGRz.pr) # slice select gradient strengths during TR

pssfp = CompasToolkit.pSSFPSequence(RF_train, TR, γΔtRF, Δt, γΔtGRz, z)
echos = CompasToolkit.simulate_magnetization(parameters, pssfp)

# isochromat model
pssfp_ref = gpu(f32(pssfp_ref))
parameters_ref = gpu(f32(parameters_ref))
echos_ref = simulate_magnetization(CUDALibs(), pssfp_ref, parameters_ref);

# Compare to compas data
print_equals_check(transpose(collect(echos_ref)), collect(echos))
