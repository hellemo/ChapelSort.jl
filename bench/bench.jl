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

# Benchmark mirroring the Chapel blog post: sort 2^27 random UInt64 (1 GiB).
#
#   julia -t auto --project=bench bench/bench.jl [log2n] [reps]
#
# Results are also written to bench/results/julia.csv (see bench/plots.jl).

using ChapelSort
import AcceleratedKernels as AK
using Printf

const LOG2N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 27
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

# The "naive" approach: sort chunks in parallel, then merge pairs in parallel rounds.
function merge_into!(dst, a, b)
    i, j, k = firstindex(a), firstindex(b), firstindex(dst)
    @inbounds while i <= lastindex(a) && j <= lastindex(b)
        if b[j] < a[i]
            dst[k] = b[j]; j += 1
        else
            dst[k] = a[i]; i += 1
        end
        k += 1
    end
    @inbounds while i <= lastindex(a); dst[k] = a[i]; i += 1; k += 1; end
    @inbounds while j <= lastindex(b); dst[k] = b[j]; j += 1; k += 1; end
    return dst
end

function chunk_merge_sort!(v, tmp = similar(v); ntasks = Threads.nthreads())
    n = length(v)
    bounds = [div((t - 1) * n, ntasks) + 1 for t in 1:(ntasks + 1)]
    bounds[end] = n + 1
    Threads.@threads for t in 1:ntasks
        sort!(view(v, bounds[t]:(bounds[t + 1] - 1)))
    end
    src, dst = v, tmp
    while length(bounds) > 2
        newb = bounds[1:2:end]
        bounds[end] in newb || push!(newb, bounds[end])
        @sync for k in 1:2:(length(bounds) - 1)
            Threads.@spawn if k + 1 < length(bounds)
                merge_into!(view(dst, bounds[k]:(bounds[k + 2] - 1)),
                            view(src, bounds[k]:(bounds[k + 1] - 1)),
                            view(src, bounds[k + 1]:(bounds[k + 2] - 1)))
            else
                copyto!(dst, bounds[k], src, bounds[k], bounds[k + 1] - bounds[k])
            end
        end
        bounds = newb
        src, dst = dst, src
    end
    src === v || copyto!(v, src)
    return v
end

const RESULTS = NamedTuple{(:method, :ntasks, :seconds), Tuple{String, Int, Float64}}[]

function bench(name, f!, x, y; method = name, ntasks = Threads.nthreads())
    copyto!(y, x); f!(y); issorted(y) || error("$name: not sorted")
    ts = Float64[]
    for _ in 1:REPS
        copyto!(y, x)
        GC.gc()
        push!(ts, @elapsed f!(y))
    end
    t = sum(ts) / length(ts)
    @printf("%-32s %8.3f s %10.1f M elem/s\n", name, t, length(x) / t / 1e6)
    push!(RESULTS, (; method, ntasks, seconds = t))
    return t
end

function main()
    n = 2^LOG2N
    nt = Threads.nthreads()
    println("Sorting 2^$LOG2N = $n random UInt64, $nt threads, mean of $REPS runs")
    println("Julia $VERSION, ", Sys.cpu_info()[1].model)
    x = rand(UInt64, n)
    y = similar(x)
    s = similar(x)
    tmp = similar(x)
    tbase = bench("Base.sort!", sort!, x, y)
    bench("chunk sort + parallel merge", v -> chunk_merge_sort!(v, tmp), x, y)
    bench("AcceleratedKernels.sort!", v -> AK.sort!(v; temp = tmp), x, y)
    nts = unique(filter(<=(nt), [1, 2, 4, 8, 16, 32, nt]))
    for k in nts
        t = bench("ChapelSort.psort! ntasks=$k", v -> psort!(v; ntasks = k, scratch = s),
                  x, y; method = "psort! (reused scratch)", ntasks = k)
        k == nt && @printf("  → %.1fx faster than Base.sort!\n", tbase / t)
        bench("ChapelSort.psort! ntasks=$k fresh", v -> psort!(v; ntasks = k), x, y;
              method = "psort! (fresh scratch)", ntasks = k)
    end
    out = joinpath(@__DIR__, "results", "julia.csv")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "method,ntasks,n,seconds")
        for r in RESULTS
            println(io, r.method, ",", r.ntasks, ",", n, ",", r.seconds)
        end
    end
    println("wrote ", out)
end

main()
