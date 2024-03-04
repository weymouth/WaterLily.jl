function parse_cla(args; cases=["tgv"], log2p=[(6,7)], max_steps=[100], ftype=[Float32], backend=Array)
    iarg(arg) = occursin.(arg, args) |> findfirst
    arg_value(arg) = split(args[iarg(arg)], "=")[end]
    metaparse(x) = eval(Meta.parse(x))

    cases = !isnothing(iarg("cases")) ? arg_value("cases") |> metaparse : cases
    log2p = !isnothing(iarg("log2p")) ? arg_value("log2p") |> metaparse : log2p
    max_steps = !isnothing(iarg("max_steps")) ? arg_value("max_steps") |> metaparse : max_steps
    ftype = !isnothing(iarg("ftype")) ? arg_value("ftype") |> metaparse : ftype
    backend = !isnothing(iarg("backend")) ? arg_value("backend") |> x -> eval(Symbol(x)) : backend
    return cases, log2p, max_steps, ftype, backend
end

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

backend_str = Dict(Array => "CPUx$(Threads.nthreads())", CuArray => "GPU")

function add_to_suite!(suite, sim_function; p=(3,4,5), s=100, ft=Float32, backend=Array)
    bstr = backend_str[backend]
    suite[bstr] = BenchmarkGroup([bstr])
    for n in p
        sim = sim_function(n, backend; T=ft)
        suite[bstr][repr(n)] = BenchmarkGroup([repr(n)])
        @add_benchmark sim_step!($sim, $typemax($ft); max_steps=$s, verbose=true, remeasure=false) $(get_backend(sim.flow.p)) suite[bstr][repr(n)] "sim_step!"
    end
end

waterlily_dir = get(ENV, "WATERLILY_ROOT", "")
git_hash = read(`git -C $waterlily_dir rev-parse --short HEAD`, String) |> x -> strip(x, '\n')
getf(str) = eval(Symbol(str))
