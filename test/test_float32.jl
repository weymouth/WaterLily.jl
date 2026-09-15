@testset "Float32 simulation on every backend (Metal has no Float64)" begin
    # Same Float32 case on the CPU and on every requested backend: fields and integrals must agree.
    # Metal cannot compile Float64, so the integrals are accumulated in Float32 there (`Ts=Float32`).
    function circle_sim(mem; T=Float32, N=64)
        L = T(N / 4); c = T(N / 2)
        body = AutoBody((x, t) -> √sum(abs2, x .- c) - L / 2)
        return Simulation((N, N), (one(T), zero(T)), L; ν=L / T(200), body, T, mem, exitBC=true)
    end
    tstep(sim) = sim_step!(sim, oftype(sim.flow.Δt[end], 1); remeasure=false)
    x₀ = SVector{2,Float32}(32, 32) # circle centre; Float32 because it is captured into the kernel

    ref = circle_sim(Array); tstep(ref)
    Fp_ref = WaterLily.pressure_force(ref); Fv_ref = WaterLily.viscous_force(ref)
    M_ref = WaterLily.total_moment(x₀, ref)
    @test Fp_ref isa Vector{Float64} && Fv_ref isa Vector{Float64} && M_ref isa Vector{Float64} # default Ts
    @test abs(Fp_ref[1]) > 0                     # drag
    @test abs(Fp_ref[2]) < 1e-2 * abs(Fp_ref[1]) # symmetric body: no lift
    # Ts is forwarded from every entry point and only changes the accumulation type
    @test WaterLily.pressure_force(ref; Ts=Float32) isa Vector{Float32}
    @test WaterLily.pressure_force(ref.flow, ref.body; Ts=Float32) ≈ Fp_ref rtol=1e-5
    @test WaterLily.total_force(ref; Ts=Float32) ≈ Fp_ref .+ Fv_ref rtol=1e-5
    @test WaterLily.total_moment(x₀, ref; Ts=Float32) ≈ M_ref atol=1e-5

    for f ∈ arrays
        sim = circle_sim(f); tstep(sim)
        @test eltype(sim.flow.u) == Float32 && eltype(sim.flow.p) == Float32 && eltype(sim.flow.μ₀) == Float32
        @test sim.flow.Δt ≈ ref.flow.Δt
        @test maximum(abs, Array(sim.flow.u) .- ref.flow.u) < 1f-4
        @test maximum(abs, Array(sim.flow.p) .- ref.flow.p) < 2f-3 # multigrid: converged to solver tolerance only
        @test maximum(abs, Array(sim.flow.μ₀) .- ref.flow.μ₀) < 1f-6
        Ts = nameof(f) == :MtlArray ? Float32 : Float64 # Metal cannot accumulate in Float64
        Fp = WaterLily.pressure_force(sim; Ts); Fv = WaterLily.viscous_force(sim; Ts)
        @test Fp isa Vector{Ts} && Fv isa Vector{Ts}
        @test Fp ≈ Fp_ref rtol=1e-4
        @test Fv ≈ Fv_ref rtol=1e-4
        @test WaterLily.total_moment(x₀, sim; Ts) ≈ M_ref atol=1e-5
    end
end
