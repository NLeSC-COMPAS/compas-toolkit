using CompasToolkit
using ImagePhantoms
include("common.jl")

context = CompasToolkit.init_context(0)

# First we assemble a Shepp Logan phantom with homogeneous T₁ and T₂
# but non-constant proton density and B₀
N = 256
ρ = ComplexF32.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())') |> vec;
T₁ = 0.3f0 .+ 1.5f0 .* rand(N, N) |> vec;     # 0.3–1.8 s
T₂ = 0.02f0 .+ 0.13f0 .* rand(N, N) |> vec;   # 0.020–0.150 s
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

nTR = N; # nr of TRs used in the simulation
RF_train = rand(nTR) .* exp.(rand(nTR) .* im) |> collect .|> complex; # flip angle train
TR,TE,TI = 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay
#max_state = 64; # maximum number of configuration states to keep track of
nz = 1
sliceprofiles = complex.(ones(nTR, nz))

# Try multiple repetitions
for max_state in [32, 64]
for repetitions in [1, 2, 6]
    TW = 0.0
    inversion_prepulse = true
    wait_spoiling = true
    undersampling_factor = 2

    # Simulate using CompasToolkit
    fisp = CompasToolkit.FispSequence(RF_train, sliceprofiles, Float32(TR), Float32(TE), max_state, Float32(TI),
                                      undersampling_factor=undersampling_factor, repetitions=repetitions)
    echos = CompasToolkit.simulate_magnetization(parameters, fisp)

    # Simulate using BlochSimulators
    fisp_ref = FISP3D(RF_train, TR, TE, Val(max_state), TI, TW, repetitions, inversion_prepulse, wait_spoiling, undersampling_factor)
    echos_ref = simulate_magnetization(CUDALibs(), gpu(f32(fisp_ref)), gpu(f32(parameters_ref)));

    # Print difference
    println("repetitions=$repetitions")
    print_equals_check(collect(echos_ref), transpose(collect(echos)))
    println()
end
end