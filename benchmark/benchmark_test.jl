using WaterLily
using BenchmarkTools
using CUDA
using KernelAbstractions: synchronize, get_backend
using StaticArrays

include("util.jl")

# cases, log2p, max_steps, ftype, backend = ["tgv"], [(8,)], [50], [Float32], Array
cases, log2p, max_steps, ftype, backend = parse_cla(ARGS;
    cases=["tgv"], log2p=[(6,)], max_steps=[100], ftype=[Float32], backend=Array
)

# Define simulation benchmarks
function tgv(p, backend; Re=1e5, T=Float32)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,vx)
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    # Initialize simulation
    return Simulation((L,L,L), (0,0,0), L; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
end

function jelly(p, backend; Re=5e2, U=1, T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    n = 2^p; R = 2n/3; h = 4n-2R; ν = U*R/Re

    # Motion functions
    ω = 2U/R
    @fastmath @inline A(t) = 1 .- SA[1,1,0]*0.1*cos(ω*t)
    @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t)-1)*R/4-h)
    @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4

    # Build jelly from a mapped sphere and plane
    sphere = AutoBody((x,t)->abs(√sum(abs2,x)-R)-1, # sdf
                      (x,t)->A(t).*x+B(t)+C(t))     # map
    plane = AutoBody((x,t)->x[3]-h,(x,t)->x+C(t))
    body =  sphere-plane

    # Return initialized simulation
    Simulation((n,n,4n), (0,0,-U), R; ν, body, T=T, mem=backend)
end

# Generate benchmarks
# function benchmark()
#     for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
#         println("Benchmarking: $(case)")
#         s = getf(case)(p[1], backend; T=ft)
#         # sim_step!(s, typemax(ft); max_steps=2, verbose=false, remeasure=false)
#         # @time sim_step!(s, typemax(ft); max_steps=10, verbose=false, remeasure=false)
#         @btime sim_step!($s, typemax($ft); max_steps=10, verbose=false, remeasure=false) samples=1 evals=1
#     end
# end

function benchmark()
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)")
        println("$(p),$(s),$(ft)")
        sim = getf(case)(p[1], backend; T=ft)
        sim_step!(sim, typemax(ft); max_steps=1, verbose=false, remeasure=false)
        CUDA.@time sim_step!(sim, typemax(ft); max_steps=s, verbose=false, remeasure=false)
        # @btime sim_step!($s, typemax($ft); max_steps=$s, verbose=false, remeasure=false) samples=1 evals=1

        # for _ in 1:10
        #     @time sim_step!(s, typemax(ft); max_steps=100, verbose=false, remeasure=false)
        # end
        # @time sim_step!(s, typemax(ft); max_steps=100, verbose=false, remeasure=false)
    end
end

benchmark()