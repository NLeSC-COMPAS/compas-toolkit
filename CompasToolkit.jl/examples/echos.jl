using CompasToolkit
using ImagePhantoms
using StructArrays

include("common.jl")

function make_phantom(N, coordinates)
    sl = shepp_logan(N, SheppLoganBrainWeb()) |> rotr90
    regions = map( val -> findall(sl .== val), unique(sl))

    T₁ = zeros(N,N)
    T₂ = zeros(N,N)
    ρˣ = zeros(N,N)
    ρʸ = zeros(N,N)

    for r in regions
        T₁[r] .= rand((0.3:0.01:2.5))
        T₂[r] .= rand((0.03:0.001:0.2))
        ρˣ[r] .= rand(0.5:0.02:1.5)
        ρʸ[r] .= rand(0.5:0.02:1.5)
    end

    T₂[ T₂ .> T₁ ] .= 0.5 * T₁[ T₂ .> T₁ ]

    return map(T₁T₂ρˣρʸ, T₁, T₂, ρˣ, ρʸ)
end

# Simulation size
N = 224; # phantom of size N^2
K = 5; # number of fully sampled Cartesian "transient-state k-spaces"
nTR = K*N; # total number of TRs

# Make sequence
RF_train = range(start=1,stop=90,length=nTR) .|> complex;
sliceprofiles = ones(nTR,1) .|> complex;
TR = 0.010;
TE = 0.006;
max_state = 32;
TI = 0.025;

# assemble sequence struct
sequence = BlochSimulators.FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI) |> f32 |> gpu;

# Make coordinates
FOVˣ = 22.4 # cm
FOVʸ = 22.4 # cm
Δx = FOVˣ/N; # cm
Δy = FOVʸ/N; # cm
x = LinRange(-FOVˣ/2, FOVˣ/2, N) # cm
y = LinRange(-FOVʸ/2, FOVʸ/2, N) # cm

coordinates = tuple.(x,y');
coordinates_ref = StructArray(Coordinates(x, y, zero(x)) for (x, y) in coordinates)

# Make trajectory
# dwell time between samples within readout
Δt = 5e-6

# phase encoding lines (linear sampling, repeated K times)
py_min = -N÷2;
py_max =  N÷2-1;
py = repeat(py_min:py_max, K);

# determine starting point in k-space for each readout
Δkˣ = 2π / FOVˣ;
Δkʸ = 2π / FOVʸ;
k_start_readout = [(-N/2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nTR];

# k-space step between samples within readout
Δk_adc = Δkˣ

# assemble trajectory struct
nreadouts = nTR
nsamplesperreadout = N

trajectory = CartesianTrajectory2D(nreadouts, nsamplesperreadout, Δt, k_start_readout, Δk_adc, py, 2)

# Make phantom
parameters = vec(make_phantom(N, coordinates));

# Make coil sensitivities
ncoils = 2
coil_sensitivities = rand((0.75:0.01:1.25), ncoils,N^2) .|> ComplexF32
coil_sensitivities .= 1
coil_sensitivities = map(SVector{ncoils}, eachcol(coil_sensitivities))

# We use two different receive coils
coil₁ = repeat(LinRange(0.8,1.2,N),1,N);
coil₂ = coil₁';
coil_sensitivities = map(SVector{2}, vec(coil₁), vec(coil₂))


# Simulate data
nvoxels = length(coordinates)
compas_context = CompasToolkit.init_context(0)
compas_sequence = CompasToolkit.FispSequence(RF_train, sliceprofiles, TR, TE, max_state, TI)
compas_parameters = CompasToolkit.TissueParameters(
    nvoxels,
    [p.T₁ for p in parameters],
    [p.T₂ for p in parameters],
    fill(1, nvoxels), # B1
    fill(0, nvoxels), # B0
    [p.ρˣ for p in parameters],
    [p.ρʸ for p in parameters],
    first.(coordinates) |> vec,
    last.(coordinates) |> vec,
)
compas_trajectory = CompasToolkit.CartesianTrajectory(
    trajectory.nreadouts,
    trajectory.nsamplesperreadout,
    trajectory.Δt,
    trajectory.k_start_readout,
    trajectory.Δk_adc[1]);

echos = CompasToolkit.simulate_magnetization(compas_parameters, compas_sequence)

# Set precision and send to gpu
parameters          = gpu(f32(vec(parameters)))
sequence            = gpu(f32(sequence))
coil_sensitivities  = gpu(f32(coil_sensitivities))
trajectory          = gpu(f32(trajectory))
coordinates_ref     = gpu(coordinates_ref)

# Simulate data
echos_ref = simulate_magnetization(CUDALibs(), sequence, parameters)

# Compare to compas data
print_equals_check(transpose(collect(echos_ref)), collect(echos))

# Phase encoding
echos = CompasToolkit.phase_encoding(echos, compas_parameters, compas_trajectory)
phase_encoding!(echos_ref, trajectory, coordinates_ref)

print_equals_check(transpose(collect(echos_ref)), collect(echos))
