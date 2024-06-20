using StaticArrays

# utilities
Base.@propagate_inbounds @inline fSV(f,n) = SA[ntuple(f,n)...]
Base.@propagate_inbounds @inline @fastmath fsum(f,n) = sum(ntuple(f,n))
norm2(x) = √(x'*x)
Base.@propagate_inbounds @fastmath function permute(f,i)
    j,k = i%3+1,(i+1)%3+1
    f(j,k)-f(k,j)
end
×(a,b) = fSV(i->permute((j,k)->a[j]*b[k],i),3)

"""
    ke(I::CartesianIndex,u,U=0)

Compute ``½∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
"""
ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = 0.125fsum(m) do i
    abs2(@inbounds(u[I,i]+u[I+δ(i,I),i]-2U[i]))
end
"""
    ∂(i,j,I,u)

Compute ``∂uᵢ/∂xⱼ`` at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
        @inbounds(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                 -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i])/4)

using LinearAlgebra: eigvals, Hermitian
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See [https://en.wikipedia.org/wiki/Lambda2_method](https://en.wikipedia.org/wiki/Lambda2_method) and
Jeong, J., & Hussain, F., doi:[10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462)
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(Hermitian(S^2+Ω^2))[2]
end

"""
    curl(i,I,u)

Compute component `i` of ``𝛁×𝐮`` at the __edge__ of cell `I`.
For example `curl(3,CartesianIndex(2,2,2),u)` will compute
`ω₃(x=1.5,y=1.5,z=2)` as this edge produces the highest
accuracy for this mix of cross derivatives on a staggered grid.
"""
curl(i,I,u) = permute((j,k)->∂(j,CI(I,k),u), i)
"""
    ω(I::CartesianIndex{3},u)

Compute 3-vector ``𝛚=𝛁×𝐮`` at the center of cell `I`.
"""
ω(I::CartesianIndex{3},u) = fSV(i->permute((j,k)->∂(k,j,I,u),i),3)
"""
    ω_mag(I::CartesianIndex{3},u)

Compute ``∥𝛚∥`` at the center of cell `I`.
"""
ω_mag(I::CartesianIndex{3},u) = norm2(ω(I,u))
"""
    ω_θ(I::CartesianIndex{3},z,center,u)

Compute ``𝛚⋅𝛉`` at the center of cell `I` where ``𝛉`` is the azimuth
direction around vector `z` passing through `center`.
"""
function ω_θ(I::CartesianIndex{3},z,center,u)
    θ = z × (loc(0,I)-SVector{3}(center))
    n = norm2(θ)
    n<=eps(n) ? 0. : θ'*ω(I,u) / n
end
"""
    ∮nds(p,body::AutoBody,t=0)

Surface normal integral of field `p` over the `body`.
"""
function ∮nds(p::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0) where {T,N}
    @loop df[I,:] .= p[I]*nds(body,loc(0,I),t) over I ∈ inside(p)
    [sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
@inline function nds(body::AbstractBody,x,t)
    d,n,_ = measure(body,x,t)
    n*WaterLily.kern(clamp(d,-1,1))
end

# Turbulence statistics
using JLD2

"""
     MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf<:AbstractArray{T}}

Holds temporal averages of velocity, squared velocity, pressure, and Reynolds stresses.
"""
struct MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf<:AbstractArray{T}}
    P :: Sf # pressure scalar field
    U :: Vf # velocity vector field
    UU :: Mf # squared velocity tensor
    τ :: Mf # Reynolds stress tensor
    t :: Vector{T} # time
    function MeanFlow(flow::Flow{D,T}; t_init=0.0) where {D,T}
        f = typeof(flow.u).name.wrapper
        P = zeros(T, size(flow.p)) |> f
        U = zeros(T, size(flow.u)) |> f
        UU = zeros(T, size(flow.p)...,D,D) |> f
        τ = zeros(T, size(UU)) |> f
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,τ,T[t_init])
    end
end
time(meanflow::MeanFlow) = meanflow.t[end]-meanflow.t[1]
function reset!(meanflow::MeanFlow; t_init=0.0)
    fill!(meanflow.P, 0); fill!(meanflow.U, 0); fill!(meanflow.UU, 0); fill!(meanflow.τ, 0)
    deleteat!(meanflow.t, collect(1:length(meanflow.t)))
    push!(meanflow.t, t_init)
end
function load!(meanflow::MeanFlow, fname::String; dir="data/")
    obj = jldopen(dir*fname)
    f = typeof(meanflow.U).name.wrapper
    meanflow.P .= obj["P"] |> f
    meanflow.U .= obj["U"] |> f
    meanflow.UU .= obj["UU"] |> f
    meanflow.τ .= obj["τ"] |> f
    meanflow.t .= obj["t"]
end
write!(fname, meanflow::MeanFlow; dir="data/") = jldsave(
    dir*fname*".jld2";
    P=Array(meanflow.P),
    U=Array(meanflow.U),
    UU=Array(meanflow.UU),
    τ=Array(meanflow.τ),
    t=meanflow.t
)

function update!(meanflow::MeanFlow, flow::Flow; stats_turb=true)
    dt = time(flow) - meanflow.t[end]
    ε = dt / (dt + (meanflow.t[end] - meanflow.t[1]) + eps(eltype(flow.p)))
    @loop meanflow.P[I] = ε*flow.p[I] + (1.0 - ε)*meanflow.P[I] over I in CartesianIndices(flow.p)
    @loop meanflow.U[Ii] = ε*flow.u[Ii] + (1.0 - ε)*meanflow.U[Ii] over Ii in CartesianIndices(flow.u)
    if stats_turb
        for i in 1:ndims(flow.p), j in 1:ndims(flow.p)
            @loop meanflow.UU[I,i,j] = ε*(flow.u[I,i].*flow.u[I,j]) + (1.0 - ε)*meanflow.UU[I,i,j] over I in CartesianIndices(flow.p)
            @loop meanflow.τ[I,i,j] = meanflow.UU[I,i,j] - meanflow.U[I,i,j]*meanflow.U[I,i,j] over I in CartesianIndices(flow.p)
        end
    end
    push!(meanflow.t, meanflow.t[end] + dt)
end
function copy!(a::Flow, b::MeanFlow)
    a.u .= b.U
    a.p .= b.P
end