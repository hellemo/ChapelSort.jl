# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Derived from the design of Chapel's standard library sort,
# modules/standard/Sort.chpl (dispatch between serial and parallel radix sorts)
# (Copyright 2020-2026 Hewlett Packard Enterprise Development LP;
# Copyright 2004-2019 Cray Inc.; Apache License 2.0).
# Modified: independent reimplementation in Julia; see NOTICE.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Public sorting entry points.

const IEEEFloat64or32 = Union{Float32,Float64}

"""
    keyspec(T, by, rev) -> Union{KeySpec,Nothing}

The radix key specification for sorting elements of type `T` by `by`, or
`nothing` when `by(x)` is not radix-sortable.
"""
function keyspec(::Type{T}, by, rev::Bool) where {T}
    KT = Base.promote_op(by, T)
    if isradixable(KT)
        kf = rev ? Rev(by) : Fwd(by)
        return FixedKey{Base.promote_op(kf, T)}(kf)
    elseif isbytestring(KT)
        return rev ? StrKey{true}(by) : StrKey{false}(by)
    end
    return nothing
end

function check_scratch(v, scratch)
    S = scratch === nothing ? similar(v) : scratch
    axes(S) == axes(v) || throw(DimensionMismatch("scratch must have the axes of v"))
    return S
end

"""
    psort!(v::AbstractVector; by=identity, rev=false, ntasks=Threads.nthreads(),
           scratch=nothing)

Sort `v` in place with a parallel two-array MSD radix sort, as used by the
Chapel standard library. `by(x)` must be radix-sortable: a bits type (see
[`isradixable`](@ref)) or a string or symbol (see [`isbytestring`](@ref));
otherwise this falls back to `sort!`. Needs one scratch array of the same size;
pass `scratch` (same axes and eltype as `v`) to reuse one.

The sort is stable.
"""
function psort!(v::AbstractVector; by = identity, rev::Bool = false,
                ntasks::Integer = Threads.nthreads(), scratch = nothing)
    length(v) < 2 && return v
    if by === identity && v isa Vector{<:IEEEFloat64or32}
        return float_psort!(v, rev, Int(ntasks), scratch)
    end
    ks = keyspec(eltype(v), by, rev)
    ks === nothing && return sort!(v; by, rev)
    return radix_sort!(v, check_scratch(v, scratch), ks, max(1, Int(ntasks)))
end

"""
    psort(v; kwargs...)

Non-mutating version of [`psort!`](@ref).
"""
psort(v::AbstractVector; kwargs...) = psort!(copy(v); kwargs...)

# --- Floats: transform to unsigned keys in place, sort, transform back. -----
#
# Computing the order-preserving key once instead of on every pass is worth
# ~30% for Float64. NaNs are moved out of the way first (to the end, or to the
# front for rev=true, as `isless` and Base do) so the key map is a bijection.

@inline function fkey(u::U) where {U<:Unsigned}
    sign = one(U) << (8 * sizeof(U) - 1)
    return u ⊻ ifelse(u & sign != 0, typemax(U), sign)
end
@inline function funkey(k::U) where {U<:Unsigned}
    sign = one(U) << (8 * sizeof(U) - 1)
    return ifelse(k & sign != 0, k ⊻ sign, ~k)
end

function float_psort!(v::Vector{F}, rev::Bool, ntasks::Int, scratch) where {F}
    S = check_scratch(v, scratch)
    n = length(v)
    nT = n < PARALLEL_CUTOFF ? 1 : max(1, ntasks)
    lo, hi = 1, n
    nnan = count(isnan, v)
    if nnan > 0
        # Stable partition through the scratch array.
        j, k = 1, n - nnan + 1
        rev && ((j, k) = (nnan + 1, 1))
        @inbounds for x in v
            if isnan(x)
                S[k] = x; k += 1
            else
                S[j] = x; j += 1
            end
        end
        copyto!(v, S)
        lo, hi = rev ? (nnan + 1, n) : (1, n - nnan)
    end
    m = hi - lo + 1
    m < 2 && return v
    U = Base.uinttype(F)
    GC.@preserve v S begin
        u = unsafe_wrap(Array, Ptr{U}(pointer(v, lo)), m)
        us = unsafe_wrap(Array, Ptr{U}(pointer(S, lo)), m)
        map_fkeys!(u, rev, nT)
        radix_sort!(u, us, FixedKey{U}(Fwd(identity)), nT)
        unmap_fkeys!(u, rev, nT)
    end
    return v
end

# (Separate functions: a closure inside float_psort! boxes the captured array.)
function map_fkeys!(u::Vector{U}, rev::Bool, nT::Int) where {U}
    foreach_block(1, length(u), nT) do _, blo, bhi
        @inbounds for i in blo:bhi
            k = fkey(u[i])
            u[i] = rev ? ~k : k
        end
    end
    return u
end

function unmap_fkeys!(u::Vector{U}, rev::Bool, nT::Int) where {U}
    foreach_block(1, length(u), nT) do _, blo, bhi
        @inbounds for i in blo:bhi
            k = u[i]
            u[i] = funkey(rev ? ~k : k)
        end
    end
    return u
end
