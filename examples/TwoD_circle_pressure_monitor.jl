using WaterLily
using StaticArrays
using LoggingExtras
include("TwoD_plots.jl")

"""Circle function"""
function circle(L=32;m=6,n=4,Re=80,U=1,T=Float32)
    radius, center = L/2, max(n*L/2,L)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((m*L,n*L), (U,0), radius; ν=U*radius/Re, body, T)
end

# make the sim
body = AutoBody((x,t)->√sum(abs2,x.-N÷2)-N÷4,(x,t)->x.-SVector(t,0))
sim = circle(64;m=24,n=16,Re=80,U=1,T=Float32)

# allows logging the pressure solver results
WaterLily.logger("test_psolver")

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1
forces_p = []; forces_ν = []

# step and plot
anim  = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # update flow
        mom_step!(sim.flow,sim.pois)
        
        # pressure force
        force = -2WaterLily.pressure_force(sim)[1]
        push!(forces_p,force)
        vforce = -2WaterLily.viscous_force(sim)[1]
        push!(forces_ν,vforce)
        # update time
        t += sim.flow.Δt[end]
    end
  
    # print time step
    println("tU/L=",round(tᵢ,digits=4),",  Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ;
    @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-10,10), legend=false); body_plot!(sim)
    contour!(sim.flow.p[inside(a)]',levels=range(-1,1,length=10),
             color=:black,linewidth=0.5,legend=false)
end
gif(anim,"cylinder.gif")

# show the pressure logger
# plot_logger("test_psolver")

# time = cumsum(sim.flow.Δt[4:end-1])
# plot(time/sim.L,forces_p[4:end]/(sim.L),label="pressure force")
# plot!(time/sim.L,forces_ν[4:end]/(sim.L),label="viscous force")
# xlabel!("tU/L"); ylabel!("force/L")