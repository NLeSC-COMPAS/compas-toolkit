using CompasToolkit
using CUDA

include("common.jl")


@inline function ∂to_sample_point(mₑ, ∂mₑ::∂mˣʸ∂T₁T₂, trajectory, readout_idx, sample_idx, p, x)
    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc

    # There are ns samples per readout, echo time is assumed to occur
    # at index (ns÷2)+1. Now compute sample index relative to the echo time
    s = sample_idx - ((ns÷2)+1)
    # Apply readout gradient, T₂ decay and B₀ rotation
    θ = Δkₓ * x
    #hasB₀(p) && ()
    θ += π*p.B₀*Δt*2
    E₂eⁱᶿ = exp(-s*Δt*R₂ + im*s*θ)

    ∂E₂eⁱᶿ = ∂mˣʸ∂T₁T₂(0, (s*Δt)*R₂*R₂*E₂eⁱᶿ)

    ∂mₛ = ∂mₑ * E₂eⁱᶿ + mₑ * ∂E₂eⁱᶿ
    mₛ = E₂eⁱᶿ * mₑ

    return mₛ, ∂mˣʸ∂T₁T₂(∂mₛ[1], ∂mₛ[2])
end

function Jv_kernel!(Jv, echos, ∂echos, parameters, coil_sensitivities::AbstractArray{SVector{Nc}{T}}, X, Y, trajectory, v) where {T,Nc}

    # global sample point index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    # sequence constants
    ns = trajectory.nsamplesperreadout # nr of samples per readout
    nr = trajectory.nreadouts # nr of readouts

    if i <= nr * ns
        r = fld1(i, ns) # determine which readout
        s = mod1(i, ns) # determine which sample within readout
        nv = length(parameters) # nr of voxels
        # v is assumed to be an array of ... SVectors?
        jv = zero(MVector{Nc, T})

        for voxel ∈ 1:nv
            x = X[voxel]

            # load coordinates, parameters, coilsensitivities and proton density for voxel
            p = parameters[voxel]
            ρ = complex(p.ρˣ,p.ρʸ)
            # x,y = coordinates[voxel]
            C = coil_sensitivities[voxel]
            # R₂ = inv(p.T₂)
            # load magnetization and partial derivatives at echo time of the r-th readout
            m  =  echos[voxel,r]
            ∂m = ∂mˣʸ∂T₁T₂(∂echos.T1[voxel,r], ∂echos.T2[voxel,r])
            # compute decay (T₂) and rotation (gradients and B₀) to go to sample point
            m, ∂m = ∂to_sample_point(m, ∂m, trajectory, r, s, p, x)
            # store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
            ∂mv = v[voxel] .* ∂mˣʸ∂T₁T₂ρˣρʸ(∂m.∂T₁, ∂m.∂T₂, m, m*im)
            for c in eachindex(C)
                lin_scale = SVector{4}(p.T₁*C[c]*ρ, p.T₂*C[c]*ρ, C[c], C[c])
                jv[c] += sum(lin_scale .* ∂mv)
            end

        end # loop over voxels

        Jv[i] = SVector(jv)
    end

    nothing
end

function compute_Jv(echos, ∂echos, parameters, coil_sensitivities::AbstractArray{SVector{Nc, T}}, X, Y, trajectory, v) where {Nc,T}
    # allocate output on GPU
    Jv = CUDA.zeros(SVector{Nc, T}, nsamples(trajectory))

    # launch cuda kernel
    THREADS_PER_BLOCK = 256
    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK Jv_kernel!(Jv, echos, ∂echos, parameters, coil_sensitivities, X, Y, trajectory, v)
    end

    return Jv
end


context = CompasToolkit.init_context(0)

# Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values
N = 256
nvoxels = N * N
T₁, T₂, B₁, B₀, ρ, X, Y = generate_parameters(N)
parameters_ref = map(T₁T₂B₀ρˣρʸ, T₁, T₂, B₀, real.(ρ), imag.(ρ))
parameters = CompasToolkit.TissueParameters(nvoxels, T₁, T₂, B₁, B₀, real.(ρ), imag.(ρ), X, Y)

# Next, we assemble a Cartesian trajectory with linear phase encoding
trajectory_ref = generate_cartesian_trajectory(N);
trajectory = CompasToolkit.CartesianTrajectory(
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

v = rand(ComplexF32, nvoxels, 4)
v_ref = map(SVector{4}, eachcol(v)...)

Jv_ref = compute_Jv(gpu(echos), gpu(∂echos), gpu(parameters_ref), gpu(coil_sensitivities_ref), gpu(X), gpu(Y), gpu(trajectory_ref), gpu(v_ref))
Jv_ref = reduce(hcat, collect(Jv_ref)) # Vector{Svector} -> Matrix

Jv = CompasToolkit.compute_jacobian(
    echos,
    ∂echos,
    parameters,
    trajectory,
    coil_sensitivities,
    v
)

print_equals_check(Jv_ref, transpose(reshape(collect(Jv), :, ncoils)))