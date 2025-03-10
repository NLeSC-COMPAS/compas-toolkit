using CompasToolkit
using CUDA
using Random

include("common.jl")

@inline function ∂expand_readout_and_accumulate_mᴴv(mᴴv, ∂mᴴv, mₑ, ∂mₑ, p, trajectory::CartesianTrajectory2D, readout_idx, t, v, x)

    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    R₂ = inv(p.T₂)

    # Gradient rotation per sample point
    θ = Δkₓ * x
    # B₀ rotation per sample point
#     θ += (hasB₀(p) ? Δt*π*p.B₀*2 : 0)
    θ += (true ? Δt*π*p.B₀*2 : 0)
    # "Rewind" to start of readout
    R = exp((ns÷2)*Δt*R₂ - im*(ns÷2)*θ)
    ∂R = ∂mˣʸ∂T₁T₂(0, -(ns÷2)*Δt*R₂*R₂*R)

    ∂mₛ = ∂mₑ * R + mₑ * ∂R
    mₛ = mₑ * R
    # T₂ decay and gradient- and B₀ induced rotation per sample
    E₂eⁱᶿ = exp(-Δt*R₂ + im*θ)
    ∂E₂eⁱᶿ = ∂mˣʸ∂T₁T₂(0, Δt*R₂*R₂*E₂eⁱᶿ)

    for sample in 1:ns
        # accumulate dot product in mᴴv
        mᴴv  += conj(mₛ)   * transpose(v[t])
        ∂mᴴv += conj(∂mₛ) .* transpose(v[t])

        # compute magnetization at next sample point
        ∂mₛ = ∂mₛ * E₂eⁱᶿ + mₛ * ∂E₂eⁱᶿ
        mₛ  = mₛ * E₂eⁱᶿ
        # increase time index
        t += 1

    end

    return mᴴv, ∂mᴴv, t
end

function Jᴴv_kernel!(Jᴴv, echos, ∂echos, parameters, coil_sensitivities::AbstractArray{SVector{Nc, T}}, X, Y, trajectory::CartesianTrajectory2D, v) where {T,Nc}
    # v is a vector of length "nr of measured samples",
    # and each element is a StaticVector of length "nr of coils"
    # output is a vector of length "nr of voxels" with each element
    voxel = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if voxel <= length(parameters)

        # sequence constants
        nr = nreadouts(trajectory) # nr of readouts

        # load parameters and spatial coordinates
        p = parameters[voxel]
        x = X[voxel]
        c = coil_sensitivities[voxel]

        # accumulators
        mᴴv = zero(SMatrix{1,Nc}{T}{Nc})
        ∂mᴴv = zero(SMatrix{2,Nc}{T}{2*Nc})

        t = 1

        for readout = 1:nr
            # load magnetization and partial derivatives at echo time of the r-th readout
            mₑ = echos[voxel,readout]
            ∂mₑ = ∂mˣʸ∂T₁T₂(∂echos.T1[voxel,readout], ∂echos.T2[voxel,readout])

            mᴴv, ∂mᴴv, t = ∂expand_readout_and_accumulate_mᴴv(mᴴv, ∂mᴴv, mₑ, ∂mₑ, p, trajectory, readout, t, v, x)
        end # loop over readouts

        tmp = vcat(∂mᴴv, mᴴv, -im*mᴴv) # size = (nr_nonlinpars + 2) x nr_coils
        Jᴴv[voxel] = zero(eltype(Jᴴv))

        ρ = complex(p.ρˣ, p.ρʸ)

        for j in eachindex(c)
            lin_scale = SVector{4}(p.T₁*c[j]*ρ, p.T₂*c[j]*ρ, c[j], c[j])
            Jᴴv[voxel] += conj(lin_scale) .* tmp[:,j]
        end
    end

    nothing
end

function compute_Jᴴv(echos::AbstractArray{T}, ∂echos, parameters, coil_sensitivities, X, Y, trajectory, v) where T
    # allocate output on GPU
    nv = length(parameters)
    Jᴴv = CUDA.zeros(∂mˣʸ∂T₁T₂ρˣρʸ{T}, nv)

    # launch cuda kernel
    THREADS_PER_BLOCK = 256
    nr_blocks = cld(nv, THREADS_PER_BLOCK)

    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK Jᴴv_kernel!(Jᴴv, echos, ∂echos, parameters, coil_sensitivities, X, Y, trajectory, v)
    end

    return Jᴴv
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

Random.seed!(1337)
v = rand(ComplexF32, trajectory_ref.nsamplesperreadout, trajectory_ref.nreadouts, ncoils)

v_ref = reshape(v, nsamples(trajectory_ref), ncoils)
v_ref = map(SVector{ncoils}, eachcol(v_ref)...)

Jᴴv_ref = compute_Jᴴv(gpu(echos), gpu(∂echos), gpu(parameters_ref), gpu(coil_sensitivities_ref), gpu(X), gpu(Y), gpu(trajectory_ref), gpu(v_ref))
Jᴴv_ref = reduce(hcat, collect(Jᴴv_ref)) # Vector{Svector} -> Matrix

Jᴴv = CompasToolkit.compute_jacobian_hermitian(
    echos,
    ∂echos,
    parameters,
    trajectory,
    coil_sensitivities,
    v
)

print_equals_check(Jᴴv_ref, transpose(collect(Jᴴv)))
