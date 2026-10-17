# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parallel stable sortperm, multi-column sortperm, and a Base.Sort.Algorithm.

# (key, index) records; the radix key is read with `GetKey`.
struct KeyIdx{K,I}
    key::K
    idx::I
end
struct GetKey end
@inline (::GetKey)(r::KeyIdx) = r.key

# Packed records for keys of at most 32 bits: key << 32 | (i - 1).
struct HiKey end
@inline (::HiKey)(x::UInt64) = (x >> 32) % UInt32

"""
    psortperm(v::AbstractVector; by=identity, rev=false, ntasks=Threads.nthreads())

Parallel stable `sortperm` built on the radix sort of [`psort!`](@ref). Falls
back to `sortperm` when `by(x)` is not radix-sortable.
"""
function psortperm(v::AbstractVector; by = identity, rev::Bool = false,
                   ntasks::Integer = Threads.nthreads())
    ks = keyspec(eltype(v), by, rev)
    ks === nothing && return sortperm(v; by, rev)
    return _psortperm(v, ks, max(1, Int(ntasks)))
end

function _psortperm(v::AbstractVector, ks::FixedKey{K}, nT::Int) where {K}
    n = length(v)
    nT = nthreads_for(n, nT)
    perm = Vector{Int}(undef, n)
    if n <= typemax(UInt32)
        sizeof(K) <= 4 && return packed_sortperm!(perm, v, ks, zero(K), nT)
        # Wider keys whose range fits in 32 bits (small integers stored in 64
        # bits, timestamps from a window, ...) are packed relative to their
        # minimum. The sample check keeps wide-range keys from paying for it.
        if n >= RANGE_PREPASS_MIN
            smin, smax = sample_extrema(v, ks)
            if smax - smin <= typemax(UInt32)
                kmin, kmax = key_extrema(v, ks, nT)
                kmax - kmin <= typemax(UInt32) &&
                    return packed_sortperm!(perm, v, ks, kmin, nT)
            end
        end
    end
    return records_sortperm!(perm, v, ks.kf, FixedKey{K}(GetKey()), nT)
end

# Sort one-word records (key - kmin) << 32 | (i - 1) by their high half; the
# radix sort's own range pre-pass turns small ranges into a counting sort.
function packed_sortperm!(perm, v, ks::FixedKey{K}, kmin::K, nT::Int) where {K}
    n = length(v)
    off = firstindex(v) - 1
    p = Vector{UInt64}(undef, n)
    foreach_block(1, n, nT) do _, lo, hi
        @inbounds for i in lo:hi
            p[i] = UInt64(ks.kf(v[i + off]) - kmin) << 32 | UInt64(i - 1)
        end
    end
    radix_sort!(p, similar(p), FixedKey{UInt32}(HiKey()), nT)
    foreach_block(1, n, nT) do _, lo, hi
        @inbounds for i in lo:hi
            perm[i] = Int(p[i] % UInt32) + 1 + off
        end
    end
    return perm
end

function _psortperm(v::AbstractVector, ks::StrKey{R}, nT::Int) where {R}
    perm = Vector{Int}(undef, length(v))
    return records_sortperm!(perm, v, ks.by, StrKey{R}(GetKey()), nthreads_for(length(v), nT))
end

function records_sortperm!(perm, v, keyof, rks::KeySpec, nT::Int)
    n = length(v)
    off = firstindex(v) - 1
    I = n <= typemax(UInt32) ? UInt32 : Int
    K = Base.promote_op(keyof, eltype(v))
    r = Vector{KeyIdx{K,I}}(undef, n)
    foreach_block(1, n, nT) do _, lo, hi
        @inbounds for i in lo:hi
            r[i] = KeyIdx{K,I}(keyof(v[i + off]), I(i))
        end
    end
    radix_sort!(r, similar(r), rks, nT)
    foreach_block(1, n, nT) do _, lo, hi
        @inbounds for i in lo:hi
            perm[i] = Int(r[i].idx) + off
        end
    end
    return perm
end

# --- Multi-column ---------------------------------------------------------

"""
    sortkeys(col, idx, rev, nT) -> (keys, rev′)

Gather the sort keys of `col[idx]` as a radix-sortable vector, to be sorted
with `rev = rev′`. Extension point: methods for column types with a cheaper
key, e.g. `PooledArray` returns the integer ranks of its sorted pool (with the
order already applied, so `rev′ = false`). Pass `Base.OneTo(length(col))` for
all rows in order.
"""
function sortkeys(col::AbstractVector, idx::AbstractVector{Int}, rev::Bool, nT::Int)
    keys = similar(col, eltype(col), length(idx))
    foreach_block(1, length(idx), nthreads_for(length(idx), nT)) do _, lo, hi
        @inbounds for i in lo:hi
            keys[i] = col[idx[i]]
        end
    end
    return keys, rev
end

function permute_gather!(dst::Vector{Int}, src::Vector{Int}, p::Vector{Int}, nT::Int)
    foreach_block(1, length(p), nthreads_for(length(p), nT)) do _, lo, hi
        @inbounds for i in lo:hi
            dst[i] = src[p[i]]
        end
    end
    return dst
end

"""
    psortperm(cols::Union{Tuple,AbstractVector{<:AbstractVector}}; rev=false, ntasks)

Stable lexicographic sortperm over several equal-length columns. `rev` is a
`Bool` or one `Bool` per column. Columns are radix sorted from last to first.
"""
function psortperm(cols::Union{Tuple,AbstractVector{<:AbstractVector}};
                   rev = false, ntasks::Integer = Threads.nthreads())
    return psortperm!(collect(1:length(first(cols))), cols; rev, ntasks)
end

"""
    psortperm!(idx::Vector{Int}, cols; rev=false, ntasks=Threads.nthreads()) -> idx

Stably reorder the row indices `idx` so that `cols` (a tuple or vector of
equal-length columns, indexed by `idx`) are in lexicographic order. `rev` is a
`Bool` or one `Bool` per column. Columns are sorted with [`sortkeys`](@ref)
keys, so pooled columns sort by rank.
"""
function psortperm!(idx::Vector{Int}, cols::Union{Tuple,AbstractVector{<:AbstractVector}};
                    rev = false, ntasks::Integer = Threads.nthreads())
    revs = rev isa Bool ? fill(rev, length(cols)) : collect(Bool, rev)
    length(revs) == length(cols) || throw(ArgumentError("need one rev per column"))
    res = multicol_sortperm!(copy(idx), Tuple(cols), revs, max(1, Int(ntasks)))
    return copyto!(idx, res)
end

# Stably reorder the indices `idx` (into the columns) by the columns.
function multicol_sortperm!(idx::Vector{Int}, cols::Tuple, revs, nT::Int)
    tmp = similar(idx)
    for j in reverse(eachindex(revs))
        keys, r = sortkeys(cols[j], idx, revs[j], nT)
        p = psortperm(keys; rev = r, ntasks = nT)
        permute_gather!(tmp, idx, p, nT)
        idx, tmp = tmp, idx
    end
    return idx
end

# --- Base.Sort algorithm ----------------------------------------------------

"""
    ChapelSortAlg()

A `Base.Sort.Algorithm` that sorts with [`psort!`](@ref)/[`psortperm`](@ref)
when the ordering allows it (Forward, Reverse, `By`, `Perm` of these), and with
Base's default stable sort otherwise. The ParallelDataFrameOps.jl package adds
support for `sort!(df, cols; alg = ChapelSortAlg())` on DataFrames.
"""
struct ChapelSortAlg <: Base.Sort.Algorithm end

using Base.Order: Ordering, ForwardOrdering, ReverseOrdering, By, Perm

"""
    orderspec(o::Base.Order.Ordering) -> Union{Tuple{Any,Bool},Nothing}

`(by, rev)` for the orderings [`ChapelSortAlg`](@ref) can radix sort (Forward,
Reverse, `By` and their combinations), or `nothing`.
"""
orderspec(::ForwardOrdering) = (identity, false)
orderspec(::ReverseOrdering{ForwardOrdering}) = (identity, true)
orderspec(o::By{<:Any,ForwardOrdering}) = (o.by, false)
orderspec(o::ReverseOrdering{<:By{<:Any,ForwardOrdering}}) = (o.fwd.by, true)
orderspec(::Ordering) = nothing

function Base.sort!(v::AbstractVector, lo::Integer, hi::Integer, ::ChapelSortAlg,
                    o::Ordering)
    spec = orderspec(o)
    spec === nothing && return sort!(v, lo, hi, Base.Sort.DEFAULT_STABLE, o)
    by, rev = spec
    psort!(view(v, lo:hi); by, rev)
    return v
end

"""
    sortable(col::AbstractVector, by) -> Bool

Whether `sortkeys`/`psortperm` can radix sort `col` by `by`. Extensions extend
this for column types with their own `sortkeys` (e.g. `PooledArray`).
"""
sortable(col::AbstractVector, by) = keyspec(eltype(col), by, false) !== nothing

function Base.sort!(v::AbstractVector{<:Integer}, lo::Integer, hi::Integer,
                    ::ChapelSortAlg, o::Perm)
    spec = orderspec(o.order)
    if spec === nothing || !sortable(o.data, spec[1])
        return sort!(v, lo, hi, Base.Sort.DEFAULT_STABLE, o)
    end
    by, rev = spec
    idx = Vector{Int}(v[lo:hi])
    nT = Threads.nthreads()
    if by === identity
        keys, rev = sortkeys(o.data, idx, rev, nT)
    else
        keys = map(i -> by(o.data[i]), idx)
    end
    p = psortperm(keys; rev, ntasks = nT)
    v[lo:hi] = idx[p]
    return v
end
