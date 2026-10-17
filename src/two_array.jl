# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Derived from the design of Chapel's standard library sort,
# modules/standard/Sort/TwoArrayPartitioning.chpl and TwoArrayRadixSort.chpl
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

# Parallel two-array partitioning sort, after Chapel's TwoArrayPartitioning module.
#
# A large range is bucketized in parallel by one digit from one array into the
# other: every task counts the buckets of its contiguous block, a bucket-major
# prefix sum hands each (bucket, task) pair a private output region, and every
# task then scatters its block without locks or atomics. Buckets that are still
# large are partitioned again (in the opposite direction); the rest become
# independent small tasks sorted serially and in parallel by `msd_sort!`.

const PARALLEL_CUTOFF = 50_000
const SEQUENTIAL_SIZE_PER_TASK = 4096

# Use one task below the cutoff, where spawning costs more than it saves.
nthreads_for(n, nT) = n < PARALLEL_CUTOFF ? 1 : nT

struct SortTask
    lo::Int
    hi::Int
    level::Int
    inA::Bool
end

@inline block(lo, n, nT, t) = (lo + div((t - 1) * n, nT), lo + div(t * n, nT) - 1)

# Run f(t, blo, bhi) for each of nT contiguous blocks of lo:hi, in parallel.
function foreach_block(f, lo::Int, hi::Int, nT::Int)
    n = hi - lo + 1
    @sync for t in 1:nT
        Threads.@spawn begin
            blo, bhi = block(lo, n, nT, t)
            f(t, blo, bhi)
        end
    end
    return nothing
end

"""
Bucketize `src[lo:hi]` into `dst[lo:hi]` by the first level, starting at
`level`, that splits the range. Returns `(level, moved)`; when `moved` is false
all keys are equal and the data was left in `src`. On return `binsizes[b]` is
the size of bucket `b`. `counts[t]` is task t's histogram/offset vector.
"""
function bucketize!(dst, src, lo, hi, level, ks::KeySpec, counts::Vector{Vector{Int}},
                    binsizes::Vector{Int}, nT)
    n = hi - lo + 1
    nb = nbuckets(ks)
    while true
        foreach_block(lo, hi, nT) do t, blo, bhi
            count_buckets!(counts[t], src, blo, bhi, level, ks)
        end
        m, bm = 0, 1
        @inbounds for b in 1:nb
            s = 0
            for t in 1:nT
                s += counts[t][b]
            end
            binsizes[b] = s
            if s > m
                m, bm = s, b
            end
        end
        m < n && break
        (!recurse(ks, bm) || level + 1 >= maxlevel(ks)) && return level, false
        level += 1
    end
    # Bucket-major exclusive prefix sum: region (b, t) follows region (b, t-1).
    pos = lo
    @inbounds for b in 1:nb, t in 1:nT
        c = counts[t][b]
        counts[t][b] = pos
        pos += c
    end
    foreach_block(lo, hi, nT) do t, blo, bhi
        scatter!(dst, src, blo, bhi, level, ks, counts[t])
    end
    return level, true
end

function run_small_tasks!(A, S, small::Vector{SortTask}, ks::KeySpec, nT)
    isempty(small) && return nothing
    # Largest first for better load balance.
    sort!(small; by = tk -> tk.hi - tk.lo, rev = true)
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:min(nT, length(small))
        Threads.@spawn begin
            counts = Vector{Int}[]
            while true
                i = Threads.atomic_add!(next, 1)
                i > length(small) && break
                tk = @inbounds small[i]
                msd_sort!(A, S, tk.lo, tk.hi, tk.level, tk.inA, ks, counts)
            end
        end
    end
    return nothing
end

function two_array_sort!(A, S, ks::KeySpec, nT::Int, level0::Int = 0)
    lo, hi = firstindex(A), lastindex(A)
    n = hi - lo + 1
    nb = nbuckets(ks)
    maxseq = max(cld(n, nT), nT * SEQUENTIAL_SIZE_PER_TASK)
    counts = [Vector{Int}(undef, nb) for _ in 1:nT]
    binsizes = Vector{Int}(undef, nb)
    big = SortTask[]
    small = SortTask[]
    push!(n > maxseq ? big : small, SortTask(lo, hi, level0, true))
    while !isempty(big)
        tk = pop!(big)
        level, moved = tk.inA ?
            bucketize!(S, A, tk.lo, tk.hi, tk.level, ks, counts, binsizes, nT) :
            bucketize!(A, S, tk.lo, tk.hi, tk.level, ks, counts, binsizes, nT)
        if !moved
            # All keys equal: only a copy back may be needed.
            tk.inA || push!(small, SortTask(tk.lo, tk.hi, DONE, false))
            continue
        end
        inA = !tk.inA
        sublevel = level + 1
        pos = tk.lo
        for b in 1:nb
            sz = binsizes[b]
            sz == 0 && continue
            sub = SortTask(pos, pos + sz - 1, sublevel, inA)
            if sz == 1 || !recurse(ks, b) || sublevel >= maxlevel(ks)
                inA || push!(small, SortTask(sub.lo, sub.hi, DONE, false))
            elseif sz > maxseq
                push!(big, sub)
            else
                push!(small, sub)
            end
            pos += sz
        end
    end
    run_small_tasks!(A, S, small, ks, nT)
    return A
end

# Sort A with scratch S (same axes) using up to nT tasks, starting at digit
# level `level0` (all keys must agree on the digits before it).
function radix_sort!(A, S, ks::KeySpec, nT::Int, level0::Int = 0)
    if length(A) < PARALLEL_CUTOFF || nT <= 1
        msd_sort!(A, S, firstindex(A), lastindex(A), level0, true, ks, Vector{Int}[])
    else
        two_array_sort!(A, S, ks, nT, level0)
    end
    return A
end

# --- Key-range pre-pass (fixed-width keys) ------------------------------------
#
# Keys confined to a small range waste work in a byte-wise MSD sort: every
# leading byte shared by all keys costs a skip-ahead counting pass, and a range
# like 0:1000 still needs two full levels. One min/max pass fixes both: a range
# of fewer than RANGE_MAX values (and fewer than n) is sorted by a single
# parallel counting sort with one bucket per value; otherwise the radix sort
# starts at the first byte where min and max differ. A strided sample decides
# first whether the pre-pass can help at all, so wide-range (e.g. uniformly
# random) keys pay only for the sample.

const RANGE_MAX = 1 << 16
const RANGE_PREPASS_MIN = 4096
const RANGE_SAMPLE = 1024

function radix_sort!(A, S, ks::FixedKey{K}, nT::Int) where {K}
    n = length(A)
    n < RANGE_PREPASS_MIN && return radix_sort!(A, S, ks, nT, 0)
    smin, smax = sample_extrema(A, ks)
    if leading_zeros(smin ⊻ smax) < 8 && smax - smin >= RANGE_MAX
        return radix_sort!(A, S, ks, nT, 0)        # wide range: pre-pass can't help
    end
    kmin, kmax = key_extrema(A, ks, nthreads_for(n, nT))
    kmin == kmax && return A                       # all keys equal
    d = kmax - kmin
    if d < min(RANGE_MAX, n)
        return counting_sort!(A, S, ks, kmin, Int(d) + 1, nthreads_for(n, nT))
    end
    return radix_sort!(A, S, ks, nT, leading_zeros(kmin ⊻ kmax) ÷ 8)
end

function sample_extrema(A, ks::FixedKey{K}) where {K}
    lo, n = firstindex(A), length(A)
    step = max(1, n ÷ RANGE_SAMPLE)
    kmin, kmax = typemax(K), typemin(K)
    @inbounds for i in lo:step:(lo + n - 1)
        k = ks.kf(A[i])
        kmin = min(kmin, k)
        kmax = max(kmax, k)
    end
    return kmin, kmax
end

function key_extrema(A, ks::FixedKey{K}, nT::Int) where {K}
    mins = fill(typemax(K), nT)
    maxs = fill(typemin(K), nT)
    foreach_block(firstindex(A), lastindex(A), nT) do t, blo, bhi
        kmin, kmax = typemax(K), typemin(K)
        @inbounds for i in blo:bhi
            k = ks.kf(A[i])
            kmin = min(kmin, k)
            kmax = max(kmax, k)
        end
        mins[t], maxs[t] = kmin, kmax
    end
    return minimum(mins), maximum(maxs)
end

# Stable parallel counting sort with r buckets for keys in kmin:kmin+r-1: the
# partition step of §3 (per-task counts, bucket-major prefix sum, scatter) with
# one bucket per key value, followed by a parallel copy back.
function counting_sort!(A, S, ks::FixedKey{K}, kmin::K, r::Int, nT::Int) where {K}
    lo, hi = firstindex(A), lastindex(A)
    counts = [zeros(Int, r) for _ in 1:nT]
    foreach_block(lo, hi, nT) do t, blo, bhi
        c = counts[t]
        @inbounds for i in blo:bhi
            c[Int(ks.kf(A[i]) - kmin) + 1] += 1
        end
    end
    pos = lo
    @inbounds for b in 1:r, t in 1:nT
        c = counts[t][b]
        counts[t][b] = pos
        pos += c
    end
    foreach_block(lo, hi, nT) do t, blo, bhi
        c = counts[t]
        @inbounds for i in blo:bhi
            x = A[i]
            b = Int(ks.kf(x) - kmin) + 1
            p = c[b]
            S[p] = x
            c[b] = p + 1
        end
    end
    foreach_block(lo, hi, nT) do _, blo, bhi
        copyto!(A, blo, S, blo, bhi - blo + 1)
    end
    return A
end
