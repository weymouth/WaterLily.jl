using WaterLily
using StaticArrays
include("TwoD_plots.jl")

function circle(n,m;Re=550,U=1,mem=Array,T=Float32)
    R, x0 = m/18, m/2+1
    bodies = AutoBody[]
    # random position x,y ∈ [-2.5,2.5] and circle diamater r ∈ [0.75,1.5]
    for (center,radius) ∈ zip(eachrow(5rand(6,2).-2.5),0.75rand(6).+0.75)
        push!(bodies,AutoBody((x,t)->√sum(abs2, x .- x0 .- 2center.*R) - radius*R))
    end
    # combine into one body
    body = Bodies(bodies, repeat([+],length(bodies)-1))
    # make a simulation
    Simulation((n,m), (U,0), R; ν=U*R/Re, body, mem, T)
end

# make a simulation and run it
sim = circle(3*2^7,2^8,mem=Array)
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)

# get force on first body
f1 = WaterLily.pressure_force(sim.flow,sim.body.bodies[1])
flood(sim.flow.f[:,:,1],clims=(-1,1)) # check that this is only non-zero near the first body

# force on the second
f2 = WaterLily.pressure_force(sim.flow,sim.body.bodies[2])
flood(sim.flow.f[:,:,1],clims=(-1,1)) # check that this is only non-zero near the first body