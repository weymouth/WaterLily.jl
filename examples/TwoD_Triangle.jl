using WaterLily
using StaticArrays
using Plots

norm(x) = √sum(abs2,x)
function make_sim(;L=2^5,U=1,Re=250,mem=Array)

    # triangle sdf    
    function triangle(p,t)
        r,k = L/2,sqrt(3.0)
        x = abs(p[1]) - r
        y = p[2] + r/k
        p = SA[clamp(x,-2r,2),y]
        if x+k*y>0.0
            p = SA[clamp(x-k*y,-2r,2),-k*x-y]./2.0
        end
        return -norm(p)*sign(p[2])
    end
    # map to center of domain
    map(x,t) = x-SA[2L,2L]

    # construct the body
    body = AutoBody(triangle,map)

    # make a simulation
    Simulation((8L,4L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end


# intialize
sim = make_sim()
# using CUDA
# sim = make_sim(mem=CuArray)
t₀,duration,tstep = sim_time(sim),10,0.1;

# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ|>Array,-10,10)',dpi=300,
             color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "doritos.gif", fps=24)