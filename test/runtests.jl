using WaterLily
import Pkg, ParallelTestRunner

#=
Parallel test runner (ParallelTestRunner.jl): every test_*.jl in this directory runs in
its own isolated module, in parallel across worker processes. Each worker gets
WATERLILY_NTHREADS threads (default 2, to exercise the multithreaded KernelAbstractions
CPU path); --jobs is auto-sized to ≈ Sys.CPU_THREADS / WATERLILY_NTHREADS unless given.

Test suite set by the WaterLily `backend` preference (LocalPreferences.toml):
    "KernelAbstractions" -> main test sets (every test_*.jl except test_alloc.jl)
    "SIMD"               -> allocation tests (test_alloc.jl) only

Array backends: WATERLILY_BACKENDS=cpu|cuda|amdgpu|metal|all (comma-separated, default "cpu").
GPU backends are opt-in and installed on demand (they are not test dependencies).
Metal has no Float64 support, so only Float32 test sets (e.g. "float32") can pass on it.

Run a subset by passing set name(s) (matched with startswith) and/or runner flags
(--jobs=N, --list, --verbose, --quickfail), e.g.
    julia --project -e 'using Pkg; Pkg.test(test_args=["poisson","flow"])'
=#

const WATERLILY_BACKENDS = filter(!isempty, strip.(split(lowercase(get(ENV, "WATERLILY_BACKENDS", "cpu")), ',')))
(isempty(WATERLILY_BACKENDS) || any(b -> b ∉ ("cpu","cuda","amdgpu","metal","all"), WATERLILY_BACKENDS)) &&
    throw(ArgumentError("WATERLILY_BACKENDS must be a comma-separated list of cpu|cuda|amdgpu|metal|all, got \"$(get(ENV, "WATERLILY_BACKENDS", ""))\""))
_cpu = any(b -> b in ("cpu","all"), WATERLILY_BACKENDS)
_cuda = any(b -> b in ("cuda","all"), WATERLILY_BACKENDS)
_amdgpu = any(b -> b in ("amdgpu","all"), WATERLILY_BACKENDS)
_metal = any(b -> b in ("metal","all"), WATERLILY_BACKENDS) && Sys.isapple()

# GPU packages are not test deps: install the requested ones once here in the main process. A backend that cannot be installed is skipped
_cuda &&
    try Pkg.add("CUDA") catch e; @warn "Requested CUDA could not be installed; skipping" exception=e; global _cuda = false end
_amdgpu &&
    try Pkg.add("AMDGPU") catch e; @warn "Requested AMDGPU could not be installed; skipping" exception=e; global _amdgpu = false end
_metal &&
    try Pkg.add("Metal") catch e; @warn "Requested Metal could not be installed; skipping" exception=e; global _metal = false end

# Discover test sets: every test_*.jl, gated by the WaterLily `backend` preference
const TESTDIR = @__DIR__
is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")
setname(f) = f[6:end-3] # "test_core.jl" -> "core"
wanted(name) = backend == "SIMD" ? name == "alloc" : name != "alloc"
testsuite = Dict{String,Expr}()
for (root, _dirs, files) in walkdir(TESTDIR), f in filter(is_test_file, files)
    name = setname(f)
    wanted(name) && (testsuite[name] = :(include($(joinpath(root, f)))))
end

# Per-sandbox setup (runs in each isolated module): imports, the `arrays` list and helpers
const init_code = quote
    using WaterLily, Test, StaticArrays
    import GPUArrays
    arrays = []
    $(_cpu)    && push!(arrays, Array)
    $(_cuda)   && using CUDA
    $(_amdgpu) && using AMDGPU
    $(_metal)  && using Metal
    $(_cuda)   && CUDA.functional()   && push!(arrays, CUDA.CuArray)
    $(_amdgpu) && AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)
    $(_metal)  && Metal.functional()  && push!(arrays, Metal.MtlArray)
    isempty(arrays) && error("No functional backend available")
    include($(joinpath(TESTDIR, "helper.jl")))
end

# WATERLILY_NTHREADS (default 2) gives every worker that many threads,
# overriding ParallelTestRunner pinning to single thread, so that tests
# run with KA multi-threading.
const _nt = get(ENV, "WATERLILY_NTHREADS", "2")
const WATERLILY_NTHREADS = isempty(_nt) ? 2 : parse(Int, _nt)
exeflags = WATERLILY_NTHREADS == 1 ? nothing : ["--threads=$WATERLILY_NTHREADS"]

# Auto-size --jobs so jobs × WATERLILY_NTHREADS ≈ the machine's thread count, unless the
# caller passed --jobs explicitly (e.g. via Pkg.test(test_args=["--jobs=N"])).
args = copy(ARGS)
any(a -> startswith(a, "--jobs"), args) ||
    push!(args, "--jobs=$(max(1, Sys.CPU_THREADS ÷ WATERLILY_NTHREADS))")

ParallelTestRunner.runtests(WaterLily, args; testsuite, init_code, exeflags)
