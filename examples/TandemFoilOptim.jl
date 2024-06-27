using WaterLily,StaticArrays

function make_foils(φ;two=true,L=64,Re=200,St=0.3,αₘ=-π/18,U=1,n=8,m=4,T=typeof(φ))
    # Map from simulation coordinate x to surface coordinate ξ
    nose,pivot = SA[2L,m*L÷2],SA[L÷4,0]
    θ₀ = T(αₘ+atan(π*St)); h₀=L; ω = T(π*St*U/h₀)
    function map(x,t)
        back = two && x[1]>nose[1]+2L # back body?
        ϕ = back ? φ : zero(φ)        # phase shift
        S = back ? 3L : zero(L)       # horizontal shift
        s,c = sincos(θ₀*cos(ω*t+ϕ))   # sin & cos of angle
        h = SA[S,h₀*sin(ω*t+ϕ)]       # position
        SA[c -s; s c]*(x-nose-h-pivot)+pivot 
    end

    # Line segment SDF
    function sdf(ξ,t)
        p = ξ-SA[clamp(ξ[1],0,L),0] # vector from closest point on [0,L] segment to ξ 
        √(p'*p)-2                   # distance (with thickness offset)
    end

    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body=AutoBody(sdf,map),T)
end

# Thrust history and mean
thrust_hist!(sim,time) = map(time) do t
    sim_step!(sim,t)
    WaterLily.pressure_force(sim)[1]
end
mean_thrust(sim,time) = sum(thrust_hist!(sim,time))/length(time)

# Optimize φ
using Optim
function two_foil_drag(φ,St=0.3)
    period = 2/St
    cost = -mean_thrust(make_foils(φ;St),range(period,2period,200))
    @show φ,cost
    return cost
end
res = optimize(x->two_foil_drag(first(x)),[2f0],Adam(alpha=0.5,beta_mean=0.5,beta_var=0.5),
        Optim.Options(iterations = 10);autodiff = :forward)