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

# Strings, symbols, InlineStrings, pooled columns, and sortperm vs Base.
#
#   julia -t 8 --project=bench bench/strings.jl [n=1e7] [reps=3]
#
# Writes bench/results/strings.csv.

using ChapelSort, InlineStrings, PooledArrays, Random, Printf

const N = length(ARGS) >= 1 ? round(Int, parse(Float64, ARGS[1])) : 10^7
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const ROWS = NamedTuple{(:case, :op, :method, :seconds), Tuple{String,String,String,Float64}}[]

function timeit(f, setup)
    f(setup())                                      # compile / warm up
    return minimum(1:REPS) do _
        x = setup()
        GC.gc()
        @elapsed f(x)
    end
end

function case(name, x)
    for (op, base, ours) in (("sort", sort!, psort!), ("sortperm", sortperm, psortperm))
        tb = timeit(base, () -> copy(x))
        tp = timeit(ours, () -> copy(x))
        @printf("%-34s %-8s Base %7.3f s   ChapelSort %7.3f s   %5.1fx\n", name, op, tb, tp, tb / tp)
        push!(ROWS, (; case = name, op, method = "Base", seconds = tb))
        push!(ROWS, (; case = name, op, method = "ChapelSort", seconds = tp))
    end
end

function main()
    Random.seed!(1)
    println("n = $N, $(Threads.nthreads()) threads, best of $REPS")
    randstr(len) = String(rand('a':'z', len))
    case("String, random, length 8–20", [randstr(rand(8:20)) for _ in 1:N])
    ids = ["id" * lpad(string(i), 10, '0') for i in 1:(N ÷ 100)]
    case("String, h2o id3 (id%010d)", rand(ids, N))
    case("String15, h2o id3", String15.(rand(ids, N)))
    case("String7, h2o id1 (100 levels)", rand(String7.(["id" * lpad(string(i), 3, '0') for i in 1:100]), N))
    case("Symbol, 10^5 distinct", rand([Symbol(randstr(rand(4:12))) for _ in 1:10^5], N))
    case("Int64", rand(Int64, N))
    case("Float64 (randn)", randn(N))
    pa = PooledArray(rand(ids, N))
    tb = timeit(sortperm, () -> pa)
    tp = timeit(psortperm, () -> pa)
    @printf("%-34s %-8s Base %7.3f s   ChapelSort %7.3f s   %5.1fx\n",
            "PooledArray{String}, 10^5 levels", "sortperm", tb, tp, tb / tp)
    push!(ROWS, (; case = "PooledArray{String}, 10^5 levels", op = "sortperm", method = "Base", seconds = tb))
    push!(ROWS, (; case = "PooledArray{String}, 10^5 levels", op = "sortperm", method = "ChapelSort", seconds = tp))
    out = joinpath(@__DIR__, "results", "strings.csv")
    open(out, "w") do io
        println(io, "case,op,method,n,nthreads,seconds")
        for r in ROWS
            println(io, "\"", r.case, "\",", r.op, ",", r.method, ",", N, ",", Threads.nthreads(), ",", r.seconds)
        end
    end
    println("wrote ", out)
end

main()
