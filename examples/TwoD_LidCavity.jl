using WaterLily
include("TwoD_plots.jl")

mag(I,u) = √sum(ntuple(i->0.25*(u[I,i]+u[I+δ(i,I),i])^2,length(I)))

# import explicitly BC function and overwrite
function BC_lid!(a,A)
    N,n = WaterLily.size_u(a)
    for j ∈ 1:n, i ∈ 1:n
        if i==1 && j==2 # lid, Dirichlet cannot be imposed, must interpolate u[i,j+1,1]+u[i,j]/2 = uBC
            @WaterLily.loop a[I,i] = 2.00-a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
            @WaterLily.loop a[I,i] = -a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
        elseif i==j # Normal direction, homoheneous Dirichlet
            @WaterLily.loop a[I,i] = 0.0 over I ∈ WaterLily.slice(N,1,j)
            @WaterLily.loop a[I,i] = 0.0 over I ∈ WaterLily.slice(N,N[j],j)
        else  # Tangential directions, interpolate ghost cell to homogeneous Dirichlet
            @WaterLily.loop a[I,i] = -a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
            @WaterLily.loop a[I,i] = -a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
        end
    end
end
@fastmath function WaterLily.mom_step!(a::Flow{N},b::AbstractPoisson) where N
    a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν,perdir=a.perdir)
    WaterLily.BDIM!(a); BC_lid!(a.u,a.U)
    WaterLily.project!(a,b); BC_lid!(a.u,a.U)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν,perdir=a.perdir)
    WaterLily.BDIM!(a); WaterLily.scale_u!(a,0.5); BC_lid!(a.u,a.U)
    WaterLily.project!(a,b,0.5); BC_lid!(a.u,a.U)
    push!(a.Δt,WaterLily.CFL(a))
end

# Initialize simulation
L = 2^7
U = 1.0
Re = 100
sim = Simulation((L,L),(0.0,0.0),L;U=U,ν=U*L/Re)
# using CUDA
# sim = Simulation((L,L),(0.0,0.0),L;U=U,ν=U*L/Re,mem=CuArray)
# get start time
t₀ = round(sim_time(sim))
duration = 20; step = 0.1

anim = @animate for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ)

    # flood plot
    @inside sim.flow.σ[I] = mag(I,sim.flow.u)
    flood(sim.flow.σ|>Array; shift=(-0.5,-0.5),clims=(0,1))

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
gif(anim,"lid_cavity.gif",fps=15)
