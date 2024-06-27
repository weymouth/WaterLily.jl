module WaterLilyAMDGPUExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using WaterLily
import WaterLily: ⋅, L₂

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

AMDGPU.allowscalar(false) # disallow scalar operations on GPU

"""
    ⋅(a,b)

Dot product of `a` and `b` `ROCArray`s reducing over a `Float64`.

[NOT TESTED]
"""
function ⋅(a::ROCArray, b::ROCArray)
    @assert size(a) == size(b) "`size(a)` and `size(b)` are not matching."
    mapreduce(+, a, b) do x, y
        promote_type(Float64,eltype(a))(x*y)
    end
end
"""
    L₂(a)

L₂ norm of ROCArray `a` excluding ghosts.
"""
L₂(a::ROCArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module
