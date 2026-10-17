# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Derived from the design of Chapel's standard library sort,
# modules/standard/Sort/MSBRadixSort.chpl
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

# Serial out-of-place MSD radix sort, used for the "small tasks" left over by the
# parallel partitioning and for whole inputs when running on one task.
#
# Data for the range `lo:hi` lives in `A` (inA == true) or `S`; on return it is
# sorted in `A`. The same range of the other array is free scratch space.
# `counts[l + 1]` is the histogram for level `l`; it grows on demand because
# string keys have no fixed depth. (Plain vectors rather than views of a
# matrix: views kept alive across the recursion would be heap allocated.)

const INSERTION_CUTOFF = 48

@inline function insertion_sort!(A, lo::Int, hi::Int, ks)
    @inbounds for i in (lo + 1):hi
        x = A[i]
        j = i - 1
        while j >= lo && keylt(ks, x, A[j])
            A[j + 1] = A[j]
            j -= 1
        end
        A[j + 1] = x
    end
    return A
end

@inline function countvec!(counts::Vector{Vector{Int}}, i::Int, nb::Int)
    while length(counts) < i
        push!(counts, Vector{Int}(undef, nb))
    end
    return @inbounds counts[i]
end

# Count buckets of src[lo:hi] at `level` into c. Returns the largest count and
# its bucket.
@inline function count_buckets!(c::Vector{Int}, src, lo::Int, hi::Int, level::Int, ks)
    fill!(c, 0)
    @inbounds for i in lo:hi
        c[bucket(ks, src[i], level)] += 1
    end
    m, bm = 0, 1
    @inbounds for b in eachindex(c)
        if c[b] > m
            m, bm = c[b], b
        end
    end
    return m, bm
end

# Stable scatter of src[lo:hi] into dst using exclusive offsets in c.
# Afterwards c[b] is one past the end of bucket b.
@inline function scatter!(dst, src, lo::Int, hi::Int, level::Int, ks, c::Vector{Int})
    @inbounds for i in lo:hi
        x = src[i]
        b = bucket(ks, x, level)
        p = c[b]
        dst[p] = x
        c[b] = p + 1
    end
    return dst
end

@inline function exclusive_prefix!(c::Vector{Int}, start::Int)
    pos = start
    @inbounds for b in eachindex(c)
        cb = c[b]
        c[b] = pos
        pos += cb
    end
    return c
end

function msd_sort!(A, S, lo::Int, hi::Int, level::Int, inA::Bool, ks::KeySpec,
                   counts::Vector{Vector{Int}})
    n = hi - lo + 1
    maxl = maxlevel(ks)
    if n <= INSERTION_CUTOFF || level >= maxl
        inA || copyto!(A, lo, S, lo, n)
        n > 1 && level < maxl && insertion_sort!(A, lo, hi, ks)
        return nothing
    end
    nb = nbuckets(ks)
    # Skip-ahead: a level where all keys share a bucket needs no data movement.
    while true
        c = countvec!(counts, level + 1, nb)
        m, bm = inA ? count_buckets!(c, A, lo, hi, level, ks) :
                      count_buckets!(c, S, lo, hi, level, ks)
        m < n && break
        if !recurse(ks, bm) || level + 1 >= maxl
            inA || copyto!(A, lo, S, lo, n)
            return nothing
        end
        level += 1
    end
    c = @inbounds counts[level + 1]
    exclusive_prefix!(c, lo)
    inA ? scatter!(S, A, lo, hi, level, ks, c) : scatter!(A, S, lo, hi, level, ks, c)
    # Recurse; children only use counts[level + 2:end], so `c` stays valid.
    s = lo
    @inbounds for b in 1:nb
        e = c[b]
        if e > s
            msd_sort!(A, S, s, e - 1, recurse(ks, b) ? level + 1 : DONE, !inA, ks, counts)
        end
        s = e
    end
    return nothing
end
