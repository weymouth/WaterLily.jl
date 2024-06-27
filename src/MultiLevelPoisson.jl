@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)
@fastmath @inline function restrict(I::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    return s
end
@fastmath @inline function restrictL(I::CartesianIndex,i,b)
    s = zero(eltype(b))
    for J ∈ up(I,i)
     s += @inbounds(b[J,i])
    end
    return 0.5s
end

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    Na = map(i->1+i÷2,N)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a::AbstractArray{T},b;perdir=()) where T
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zeros(SVector{n,T}),false,perdir)  # correct μ₀ @ boundaries
end
restrict!(a,b) = @inside a[I] = restrict(I,b)
prolongate!(a,b) = @inside a[I] = b[down(I)]

@inline divisible(N) = mod(N,2)==0 && N>4
@inline divisible(l::Poisson) = all(size(l.x) .|> divisible)
"""
    MultiLevelPoisson{N,M}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The only variable is `levels`, a vector of nested `Poisson` systems.
"""
struct MultiLevelPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    x::S
    L::V
    z::S
    levels :: Vector{Poisson{T,S,V}}
    n :: Vector{Int16}
    perdir :: NTuple # direction of periodic boundary condition
    maxlevels :: Vector{Int16} # wrap in vector so it can be updated
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};maxlevels=Inf,perdir=()) where T
        levels = Poisson[Poisson(x,L,z;perdir)]
        while divisible(levels[end]) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],perdir,[length(levels)])
    end
end

function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

function Vcycle!(ml::MultiLevelPoisson;l=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (with recursion if possible)
    l+1<ml.maxlevels[1] && Vcycle!(ml,l=l+1)
    smooth!(coarse)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine)
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)

function solver!(ml::MultiLevelPoisson;tol=2e-4,itmx=32)
    p = ml.levels[1]
    residual!(p); r₂ = L∞(p)
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        Vcycle!(ml)
        smooth!(p); r₂ = L∞(p)
        nᵖ+=1
    end
    perBC!(p.x,p.perdir)
    (nᵖ<2 && ml.maxlevels[1]>5) && (ml.maxlevels[1]-=1) # remove coarsest level if this was easy
    (nᵖ>4 && ml.maxlevels[1]<length(ml.levels)) && (ml.maxlevels[1]+=1) # add a level if this was hard
    push!(ml.n,nᵖ);
end
