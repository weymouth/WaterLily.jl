"""
    Poisson{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover,
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`. This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the Poisson structure holds
helper arrays for `inv(D)`, the error `ϵ`, and residual `r=z-Ax`. An iterative
solution method then estimates the error `ϵ=̃A⁻¹r` and increments `x+=ϵ`, `r-=Aϵ`.
"""
abstract type AbstractPoisson{T,S,V} end
struct Poisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    L :: V # Lower diagonal coefficients
    D :: S # Diagonal coefficients
    iD :: S # 1/Diagonal
    x :: S # approximate solution
    ϵ :: S # increment/error
    r :: S # residual
    z :: S # source
    n :: Vector{Int16} # pressure solver iterations
    perdir :: NTuple # direction of periodic boundary condition
    function Poisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        set_diag!(D,iD,L)
        new{T,typeof(x),typeof(L)}(L,D,iD,x,ϵ,r,z,[],perdir)
    end
end

using ForwardDiff: Dual,Tag
Base.eps(::Type{D}) where D<:Dual{Tag{G,T}} where {G,T} = eps(T)
function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    @inside iD[I] = abs2(D[I])<2eps(eltype(D)) ? 0. : inv(D[I])
end
update!(p::Poisson) = set_diag!(p.D,p.iD,p.L)

@fastmath @inline function diag(I::CartesianIndex{d},L) where {d}
    s = zero(eltype(L))
    for i in 1:d
        s -= @inbounds(L[I,i]+L[I+δ(i,I),i])
    end
    return s
end

"""
    mult!(p::Poisson,x)

Efficient function for Poisson matrix-vector multiplication.
Fills `p.z = p.A x` with 0 in the ghost cells.
"""
function mult!(p::Poisson,x)
    @assert axes(p.z)==axes(x)
    perBC!(x,p.perdir)
    fill!(p.z,0)
    @inside p.z[I] = mult(I,p.L,p.D,x)
    return p.z
end
@fastmath @inline function mult(I::CartesianIndex{d},L,D,x) where {d}
    s = @inbounds(x[I]*D[I])
    for i in 1:d
        s += @inbounds(x[I-δ(i,I)]*L[I,i]+x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s
end

"""
    residual!(p::Poisson)

Computes the resiual `r = z-Ax` and corrects it such that
`r = 0` if `iD==0` which ensures local satisfiability
    and
`sum(r) = 0` which ensures global satisfiability.

The global correction is done by adjusting all points uniformly,
minimizing the local effect. Other approaches are possible.

Note: These corrections mean `x` is not strictly solving `Ax=z`, but
without the corrections, no solution exists.
"""
function residual!(p::Poisson)
    perBC!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-mult(I,p.L,p.D,p.x))
    s = sum(p.r)/length(inside(p.r))
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

function increment!(p::Poisson)
    perBC!(p.ϵ,p.perdir)
    @loop (p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ);
           p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ inside(p.x)
end
"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother run `it` times.
Note: This runs for general backends, but is _very_ slow to converge.
"""
@fastmath Jacobi!(p;it=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    increment!(p)
end

"""
    pcg!(p::Poisson; it=6, f=2)

Conjugate-Gradient smoother with Jacobi preditioning. Runs at most `it` iterations,
but will exit early if the step-size `|α| < 1%` or residual `ρ = r'D⁻¹r < 10eps`.
Will also exit if `ρᵏ⁺¹ ≥ fρᵏ` to avoid divergence.
"""
function pcg!(p::Poisson{T};it=6,f=2) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    abs(rho)<10eps(T) && return
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(z⋅ϵ)
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        (i==it || abs(alpha)<1e-2) && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        (abs(rho2)<10eps(T) || f*abs(rho)<abs(rho2)) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end
smooth!(p) = pcg!(p,f=1.1) # fail faster

L₂(p::Poisson) = p.r ⋅ p.r # special method since outside(p.r)≡0
L∞(p::Poisson) = maximum(abs,p.r)

"""
    solver!(A::Poisson;log,tol,itmx)

Approximate iterative solver for the Poisson matrix equation `Ax=b`.

  - `A`: Poisson matrix with working arrays.
  - `A.x`: Solution vector. Can start with an initial guess.
  - `A.z`: Right-Hand-Side vector. Will be overwritten!
  - `A.n[end]`: stores the number of iterations performed.
  - `log`: If `true`, this function returns a vector holding the `L₂`-norm of the residual at each iteration.
  - `tol`: Convergence tolerance on the `L₂`-norm residual.
  - `itmx`: Maximum number of iterations.
"""
function solver!(p::Poisson;log=false,tol=1e-4,itmx=1e3)
    residual!(p); r₂ = L₂(p)
    log && (res = [r₂])
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        pcg!(p); r₂ = L₂(p)
        log && push!(res,r₂)
        nᵖ+=1
    end
    perBC!(p.x,p.perdir)
    push!(p.n,nᵖ)
    log && return res
end