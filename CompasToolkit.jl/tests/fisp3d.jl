using Pkg
Pkg.activate("CompasToolkit.jl/tests")

using Revise
using BlochSimulators
using CompasToolkit
using ImagePhantoms
using ComputationalResources
using LinearAlgebra
using StaticArrays
using CUDA

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
parameters = CompasToolkit.make_tissue_parameters(context, nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

nTR = 5N; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect .|> complex; # flip angle train
TR,TE,TI = 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay
max_state = 32; # maximum number of configuration states to keep track of
nz = 35
sliceprofiles = complex.(ones(nTR, nz))

fisp_ref = FISP(RF_train, sliceprofiles, TR, TE, max_state, TI);

TW = 0.1
repetitions = 5
inversion_prepulse = true
wait_spoiling = true

fisp3d_ref = BlochSimulators.FISP3D(RF_train[1:nTR], TR, TE, max_state, TI, TW, repetitions, inversion_prepulse, wait_spoiling) |> f32 |> gpu;

RF_train = RF_train .|> ComplexF32 # constant flip angle train
sliceprofiles = collect(sliceprofiles)  .|> ComplexF32 # z locations

fisp = CompasToolkit.make_fisp_sequence(context, RF_train, sliceprofiles, Float32(TR), Float32(TE), max_state, Float32(TI))

fisp_ref = gpu(f32(fisp_ref))
parameters_ref = gpu(f32(parameters_ref))

# ## test fisp

# # isochromat model
# @time echos_ref = simulate_echos(CUDALibs(), fisp_ref, parameters_ref);

# echos_ref = collect(echos_ref)

# @time echos = zeros(ComplexF32, nvoxels, nTR)
# @time CompasToolkit.simulate_sequence(context, echos, parameters, fisp)

# @time begin
# d_echos = CUDA.zeros(ComplexF32, nvoxels, nTR)
# CompasToolkit.simulate_sequence(context, d_echos, parameters, fisp)
# d_echos[1]
# end

# Array(d_echos) == echos

# echos = transpose(echos)

# println("fraction equal: ", sum(echos .≈ echos_ref) / length(echos))

# err = abs.(echos - echos_ref)
# println("maximum abs error: ", maximum(err))
# println("maximum rel error: ", maximum(a / b for (a, b) in zip(err, abs.(echos_ref)) if b != 0))

## test fisp3d

fisp3d = CompasToolkit.make_fisp3d_sequence(context, RF_train[1:nTR÷2], Float32(TR), Float32(TE), max_state, Float32(TI), Float32(TW))

@time echos_ref = simulate_echos(CUDALibs(), fisp3d_ref, parameters_ref);

@time begin
    echos = CUDA.zeros(ComplexF32, nvoxels, nTR)
    CompasToolkit.simulate_sequence(context, echos, parameters, fisp3d)
    echos[1]
end;

# err = abs.(echos - echos_ref)
# println("maximum abs error: ", maximum(err))
# println("maximum rel error: ", maximum(a / b for (a, b) in zip(collect(err), collect(abs.(echos_ref))) if b != 0))
