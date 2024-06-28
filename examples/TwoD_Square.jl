using WaterLily
using StaticArrays
using Plots

function make_sim(;L=2^5,U=1,Re=250,mem=Array)

    # plane sdf
    function plane(x,t,center,normal)
        normal = normal/√sum(abs2,normal)
        sum((x .- center).*normal)
    end
    # map to center of domain
    map(x,t) = x-SA[2L,2L]

    # square is intersection of four planes
    body = AutoBody((x,t)->plane(x,t,SA[-L/2,0],SA[-1, 0]),map) ∩ 
           AutoBody((x,t)->plane(x,t,SA[ 0,L/2],SA[ 0, 1]),map) ∩ 
           AutoBody((x,t)->plane(x,t,SA[L/2, 0],SA[ 1, 0]),map) ∩ 
           AutoBody((x,t)->plane(x,t,SA[0,-L/2],SA[ 0,-1]),map)

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
gif(anim, "square.gif", fps=24)