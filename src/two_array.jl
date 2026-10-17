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

function two_array_sort!(A, S, ks::KeySpec, nT::Int)
    lo, hi = firstindex(A), lastindex(A)
    n = hi - lo + 1
    nb = nbuckets(ks)
    maxseq = max(cld(n, nT), nT * SEQUENTIAL_SIZE_PER_TASK)
    counts = [Vector{Int}(undef, nb) for _ in 1:nT]
    binsizes = Vector{Int}(undef, nb)
    big = SortTask[]
    small = SortTask[]
    push!(n > maxseq ? big : small, SortTask(lo, hi, 0, true))
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

# Sort A with scratch S (same axes) using up to nT tasks.
function radix_sort!(A, S, ks::KeySpec, nT::Int)
    if length(A) < PARALLEL_CUTOFF || nT <= 1
        msd_sort!(A, S, firstindex(A), lastindex(A), 0, true, ks, Vector{Int}[])
    else
        two_array_sort!(A, S, ks, nT)
    end
    return A
end
