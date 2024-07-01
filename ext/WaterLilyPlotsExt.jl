module WaterLilyPlotsExt

if isdefined(Base, :get_extension)
    using Plots; gr()
else
    using ..Plots; gr()
end

using WaterLily
import WaterLily: flood,addbody,body_plot!,sim_gif!,plot_logger

"""
    flood(f)

Plot a filled contour plot of the 2D array `f`. The keyword arguments are passed to `Plots.contourf`.
"""
function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f'|>Array,
                   linewidth=0, levels=levels, color=cfill, clims = clims, 
                   aspect_ratio=:equal; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(sim.flow.σ[R]'|>Array;levels,lines)
end

"""
    sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)

Make a gif of the simulation `sim` for `duration` seconds with `step` time steps. The keyword arguments are passed to `flood` and `body_plot!`.
"""
function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R]; kv...)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end


"""
    plot_logger(fname="WaterLily.log")

Plot the residuals and MG iterations from the log file `fname`.
"""
function plot_logger(fname="WaterLily.log")
    predictor = []; corrector = []
    open(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log","r") do f
        readline(f) # read first line and dump it
        which = "p"
        while ! eof(f)  
            s = split(readline(f) , ",")          
            which = s[1] != "" ? s[1] : which
            push!(which == "p" ? predictor : corrector,parse.(Float64,s[2:end]))
        end
    end
    predictor = reduce(hcat,predictor)
    corrector = reduce(hcat,corrector)

    # get index of all time steps
    idx = findall(==(0.0),@views(predictor[1,:]))
    # plot inital L∞ and L₂ norms of residuals for the predictor step
    p1=plot(1:length(idx),predictor[2,idx],color=:1,ls=:dash,label="predictor initial r∞",yaxis=:log,size=(800,400),dpi=600,
            xlabel="Time step",ylabel="L∞-norm",title="Residuals",ylims=(1e-6,1e0),xlims=(0,length(idx)))
    p2=plot(1:length(idx),predictor[2,idx],color=:1,ls=:dash,label="predictor initial r₂",yaxis=:log,size=(800,400),dpi=600,
            xlabel="Time step",ylabel="L₂-norm",title="Residuals",ylims=(1e-6,1e0),xlims=(0,length(idx)))
    # plot final L∞ and L₂norms of residuals for the predictor
    plot!(p1,1:length(idx),vcat(predictor[2,idx[2:end].-1],predictor[2,end]),color=:1,lw=2,label="predictor r∞")
    plot!(p2,1:length(idx),vcat(predictor[3,idx[2:end].-1],predictor[3,end]),color=:1,lw=2,label="predictor r₂")
    # plot the MG iterations for the predictor
    p3=plot(1:length(idx),clamp.(vcat(predictor[1,idx[2:end].-1],predictor[1,end]),√1/2,32),lw=2,label="predictor",size=(800,400),dpi=600,
            xlabel="Time step",ylabel="Iterations",title="MG Iterations",ylims=(√1/2,32),xlims=(0,length(idx)),yaxis=:log2)
    yticks!([√1/2,1,2,4,8,16,32],["0","1","2","4","8","16","32"])
    # get index of all time steps
    idx = findall(==(0.0),@views(corrector[1,:]))
    # plot inital L∞ and L₂ norms of residuals for the corrector step
    plot!(p1,1:length(idx),corrector[2,idx],color=:2,ls=:dash,label="corrector initial r∞",yaxis=:log)
    plot!(p2,1:length(idx),corrector[3,idx],color=:2,ls=:dash,label="corrector initial r₂",yaxis=:log)
    # plot final L∞ and L₂ norms of residuals for the corrector step
    plot!(p1,1:length(idx),vcat(corrector[2,idx[2:end].-1],corrector[2,end]),color=:2,lw=2,label="corrector r∞")
    plot!(p2,1:length(idx),vcat(corrector[3,idx[2:end].-1],corrector[3,end]),color=:2,lw=2,label="corrector r₂")
    # plot MG iterations of the corrector
    plot!(p3,1:length(idx),clamp.(vcat(corrector[1,idx[2:end].-1],corrector[1,end]),√1/2,32),lw=2,label="corrector")
    # plot all together
    plot(p1,p2,p3,layout=@layout [a b c])
end

end # module