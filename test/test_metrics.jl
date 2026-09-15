import WaterLily: ×
@testset "Metrics.jl" begin
    J = CartesianIndex(2,3,4); x = loc(0,J,Float64); px = prod(x)
    @test WaterLily.shiftDir(1,3,2) == 3
    @test WaterLily.shiftDir(1,4,-1) == 4
    @test WaterLily.shiftDir(4,4,1) == 1
    for f ∈ arrays
        u = zeros(3,4,5,3) |> f; apply!((i,x)->x[i]+prod(x),u)
        p = zeros(3,4,5) |> f
        @inside p[I] = WaterLily.ke(I,u)
        @test GPUArrays.@allowscalar p[J]==0.5*sum(abs2,x .+ px)
        @inside p[I] = WaterLily.ke(I,u,x)
        @test GPUArrays.@allowscalar p[J]==1.5*px^2
        @inside p[I] = WaterLily.λ₂(I,u)
        @test GPUArrays.@allowscalar p[J]≈1
        ω = (1 ./ x)×repeat([px],3)
        @inside p[I] = WaterLily.curl(2,I,u)
        @test GPUArrays.@allowscalar p[J]==ω[2]
        f==Array && @test WaterLily.ω(J,u)≈ω
        @inside p[I] = WaterLily.ω_mag(I,u)
        @test GPUArrays.@allowscalar p[J]==sqrt(sum(abs2,ω))
        @inside p[I] = WaterLily.ω_θ(I,(0,0,1),x .+ (0,1,2),u)
        @test GPUArrays.@allowscalar p[J]≈ω[1]
        # test helicity
        u_h = zeros(6,6,6,3) |> f; apply!((i,x)-> i==1 ? x[1] : 0.0, u_h)
        ω_h = zeros(6,6,6,3) |> f; apply!((i,x)-> i==1 ? x[2]-0.5 + 1 : 0.0, ω_h)
        I_h = CartesianIndex(3,3,3)
        T_h = eltype(u_h)
        umid = loc(0,I_h,T_h)[1]; ωmid=loc(0,I_h,T_h)[2]+1
        @test GPUArrays.@allowscalar WaterLily.helicity(I_h,u_h,ω_h) == umid*ωmid
        
        apply!((x)->1,p)
        @test WaterLily.L₂(p)≈prod(size(p).-2)
        # test force routines
        N = 32
        p = zeros(N,N) |> f; df₂ = zeros(N,N,2) |> f; df₃ = zeros(N,N,N,3) |> f
        @inside p[I] = loc(0, I, eltype(p))[2]
        body = AutoBody((x,t)->√sum(abs2,x.-(N/2))-N÷4,(x,t)->x)
        force = WaterLily.pressure_force(p,df₂,body)
        @test sum(abs,force/(π*(N/4)^2) - [0,1]) < 2e-3
        @test WaterLily.pressure_force(p,df₂,body;Ts=Float32) ≈ force # Float64 fields still accumulate in Float64
        # stress tensor
        u₂ = zeros(N,N,2) |> f
        u₃ = zeros(N,N,N,3) |> f
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ 0)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ 0)
        apply!((i,x)->x[i],u₂) # uniform gradient
        apply!((i,x)->x[i],u₃)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[2 0; 0 2])
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[2 0 0; 0 2 0; 0 0 2])
        apply!((i,x)->x[i%2+1],u₂) # shear
        apply!((i,x)->x[i%3+1],u₃)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[0 2; 2 0])
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[0 1 1; 1 0 1; 1 1 0])
        # viscous force
        u₂ .= 0; u₃ .= 0
        @test all(WaterLily.viscous_force(u₂,1.0,df₂,body) .≈ 0)
        @test all(WaterLily.viscous_force(u₃,1.0,df₃,body) .≈ 0)
        # viscous moment
        u₂ .= 0; u₃ .= 0
        @test all(WaterLily.viscous_moment(SVector{2,Float64}(N/2,N/2),u₂,1.0,df₂,body) .≈ 0)
        @test all(WaterLily.viscous_moment(SVector{3,Float64}(N/2,N/2,N/2),u₃,1.0,df₃,body) .≈ 0)
        # pressure moment
        p₂ = zeros(N,N) |> f; apply!(x->x[2],p₂)
        p₃ = zeros(N,N,N) |> f; apply!(x->x[2],p₃)
        @test WaterLily.pressure_moment(SVector{2,Float64}(N/2,N/2),p₂,df₂,body,0)[1] ≈ 0 # no moment in hydrostatic pressure
        @test all(WaterLily.pressure_moment(SVector{3,Float64}(N/2,N/2,N/2),p₃,df₃,body,0) .≈ SA[0 0 0]) # with a 3D field, 3D moments
        # temporal averages
        T = Float32
        sim = make_bl_flow(; T, mem=f)
        meanflow = MeanFlow(sim.flow; uu_stats=true)
        for t in range(0,10;step=0.1)
            sim_step!(sim, t)
            update!(meanflow, sim.flow)
        end
        @test all(isapprox.(Array(sim.flow.u), Array(meanflow.U); atol=√eps(T))) # can't broadcast isapprox for GPUArrays...
        @test all(isapprox.(Array(sim.flow.p), Array(meanflow.P); atol=√eps(T)))
        for i in 1:ndims(sim.flow.p), j in 1:ndims(sim.flow.p)
            @test all(isapprox.(Array(sim.flow.u)[:,:,i] .* Array(sim.flow.u)[:,:,j], Array(meanflow.UU)[:,:,i,j]; atol=√eps(T)))
        end
        τ = uu(meanflow)
        for i in 1:ndims(sim.flow.p), j in 1:ndims(sim.flow.p)
            @test all(isapprox.(
                Array(meanflow.UU)[:,:,i,j] .- Array(meanflow.U)[:,:,i].*Array(meanflow.U)[:,:,j],
                Array(τ)[:,:,i,j]; atol=√eps(T))
            )
        end
        @test WaterLily.time(sim.flow) == WaterLily.time(meanflow)
        WaterLily.reset!(meanflow)
        @test all(meanflow.U .== zero(T))
        @test all(meanflow.P .== zero(T))
        @test all(meanflow.UU .== zero(T))
        @test meanflow.t == T[0]

        meanflow2 = MeanFlow(size(sim.flow.p).-2; uu_stats=true)
        @test all(meanflow2.P .== zero(T))
        @test size(meanflow2.P) == size(meanflow.P)
    end
end
