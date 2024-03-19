using BlochSimulators
using CompasToolkit
using ImagePhantoms
using ComputationalResources
using LinearAlgebra
using StaticArrays

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
parameters_ref = map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y)
parameters = CompasToolkit.TissueParameters(nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

nTR = N; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect .|> complex; # flip angle train
TR,TE,TI = 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay
max_state = 25; # maximum number of configuration states to keep track of
nz = 35
sliceprofiles = complex.(ones(nTR, nz))

fisp_ref = FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI);
RF_train = RF_train .|> ComplexF32 # constant flip angle train
sliceprofiles = collect(sliceprofiles)  .|> ComplexF32 # z locations

# Simulate data
fisp = CompasToolkit.FispSequence(RF_train, sliceprofiles, Float32(TR), Float32(TE), max_state, Float32(TI))
echos = CompasToolkit.simulate_magnetization(parameters, fisp)

# isochromat model
fisp_ref = gpu(f32(fisp_ref))
parameters_ref = gpu(f32(parameters_ref))
echos_ref = simulate_magnetization(CUDALibs(), fisp_ref, parameters_ref);

print_equals_check(collect(echos_ref), transpose(collect(echos)))
