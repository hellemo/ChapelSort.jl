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

using ChapelSort
using Test
using InlineStrings, PooledArrays

shuffle_ids(n) = sortperm(rand(n))

const SIZES = [0, 1, 2, 15, 16, 17, 47, 48, 49, 1000, 49_999, 50_000, 300_000]

function gen(::Type{T}, n, dist) where {T}
    dist === :uniform && return rand(T, n)
    dist === :equal && return fill(rand(T), n)
    dist === :few && return rand(rand(T, 5), n)
    dist === :sorted && return sort!(rand(T, n))
    dist === :reverse && return sort!(rand(T, n); rev = true)
    if dist === :lowbits
        T <: Integer || return rand(T, n)
        base = rand(T)
        return [base ⊻ (rand(UInt8) % T) for _ in 1:n]
    end
    error("unknown $dist")
end

@testset "ChapelSort" begin
    @testset "radixkey order" begin
        xs = [-Inf, -1.0, -floatmin(Float64), -nextfloat(0.0), -0.0, 0.0,
              nextfloat(0.0), 1.0, Inf, NaN, -NaN]
        for a in xs, b in xs
            isequal(a, b) && continue
            @test isless(a, b) == (ChapelSort.radixkey(a) < ChapelSort.radixkey(b))
        end
    end

    @testset "$T" for T in (UInt8, UInt16, UInt32, UInt64, UInt128, Int8, Int16,
                            Int32, Int64, Int128, Float32, Float64, Char, Bool)
        for n in SIZES, dist in (:uniform, :equal, :few, :sorted, :reverse, :lowbits)
            T === Char && dist !== :uniform && continue
            x = T === Char ? rand(Char, n) : gen(T, n, dist)
            @test isequal(psort(x), sort(x))
            @test isequal(psort(x; rev = true), sort(x; rev = true))
        end
    end

    @testset "special floats" begin
        sp = [NaN, -NaN, Inf, -Inf, 0.0, -0.0, floatmin(Float64), -floatmin(Float64),
              nextfloat(0.0), -nextfloat(0.0)]
        for n in (20, 100_000)
            x = [rand() < 0.2 ? rand(sp) : randn() for _ in 1:n]
            @test isequal(psort(x), sort(x))
            @test isequal(psort(x; rev = true), sort(x; rev = true))
        end
    end

    @testset "by / stability" begin
        x = [(rand(1:50), i) for i in 1:200_000]
        @test psort(x; by = first) == sort(x; by = first)       # stable
        @test psort(x; by = first, rev = true) == sort(x; by = first, rev = true)
    end

    @testset "ntasks and scratch" begin
        x = rand(Int64, 1_000_000)
        s = similar(x)
        for nt in (1, 2, 3, 8)
            @test psort(x; ntasks = nt, scratch = s) == sort(x)
        end
    end

    @testset "strings and symbols" begin
        alphabet = ['a', 'b', 'c', '\0', 'é', '∀']
        randstr(maxlen) = String(rand(alphabet, rand(0:maxlen)))
        invalid = [String(rand(UInt8, rand(0:6))) for _ in 1:200]
        for n in (0, 1, 2, 47, 48, 49, 1000, 49_999, 50_000, 300_000)
            for x in ([randstr(4) for _ in 1:n],                       # many ties
                      [randstr(12) for _ in 1:n],
                      ["id" * lpad(string(rand(1:10^6)), 10, '0') for _ in 1:n],  # shared prefix
                      fill("same", n),
                      [rand(invalid) for _ in 1:n])
                @test isequal(psort(x), sort(x))
                @test isequal(psort(x; rev = true), sort(x; rev = true))
            end
            s = [SubString(randstr(10), 1) for _ in 1:n]
            @test isequal(psort(s), sort(s))
            y = [Symbol(replace(randstr(6), '\0' => 'z')) for _ in 1:n]
            @test isequal(psort(y), sort(y))
            @test isequal(psort(y; rev = true), sort(y; rev = true))
        end
        x = [(randstr(5), i) for i in 1:200_000]
        @test psort(x; by = first) == sort(x; by = first)      # stable
        @test psort(x; by = first, rev = true) == sort(x; by = first, rev = true)
    end

    @testset "float NaN partition" begin
        for n in (10, 100_000), F in (Float32, Float64)
            x = F[rand() < 0.1 ? (rand(Bool) ? F(NaN) : -F(NaN)) : randn(F) for _ in 1:n]
            @test isequal(psort(x), sort(x))
            @test isequal(psort(x; rev = true), sort(x; rev = true))
            @test isequal(psort(x; ntasks = 1), sort(x))
        end
    end

    @testset "fallback" begin
        x = [BigInt(rand(1:1000)) for _ in 1:1000]
        @test psort(x) == sort(x)
        y = view(rand(UInt32, 100_000), 1:2:100_000)
        @test psort!(copy(y)) == sort(y)
    end

    @testset "psortperm" begin
        for n in (0, 1, 47, 49, 1000, 60_000, 300_000)
            for x in (rand(Int8, n), rand(Int32, n), rand(UInt64, n), randn(n),
                      rand(1:10, n), [String(rand('a':'c', rand(0:4))) for _ in 1:n],
                      [Symbol(rand('a':'c', 3)...) for _ in 1:n], rand(Char, n))
                @test psortperm(x) == sortperm(x)                 # stable ⇒ identical
                @test psortperm(x; rev = true) == sortperm(x; rev = true)
            end
        end
        x = randn(100_000)
        @test psortperm(x; by = abs) == sortperm(x; by = abs)
        v = view(rand(1:100, 200_000), 2:2:200_000)
        @test psortperm(v) == sortperm(v)
        a, b, c = rand(1:5, 200_000), [String(rand('a':'d', 2)) for _ in 1:200_000], randn(200_000)
        p = sortperm(collect(zip(a, b, c)))
        @test psortperm((a, b, c)) == p
        @test psortperm((a, b); rev = [true, false]) ==
              sortperm(collect(zip(a, b)); lt = (x, y) -> x[1] != y[1] ? x[1] > y[1] : x[2] < y[2])
    end

    @testset "InlineStrings" begin
        for T in (String1, String3, String7, String15, String31, String63), n in (100, 100_000)
            maxlen = min(sizeof(T) - 1, 20)
            x = [T(String(rand(['a', 'b', '\0', 'z'], rand(0:maxlen)))) for _ in 1:n]
            @test isequal(psort(x), sort(x))
            @test isequal(psort(x; rev = true), sort(x; rev = true))
            @test psortperm(x) == sortperm(x)
        end
        x = [String15("id" * lpad(string(rand(1:10^5)), 10, '0')) for _ in 1:200_000]
        @test psort(x) == sort(x)
    end

    @testset "PooledArrays" begin
        for n in (100, 200_000)
            pa = PooledArray([String(rand('a':'z', 3)) for _ in 1:n])
            @test psortperm(pa) == sortperm(collect(pa))
            @test psortperm(pa; rev = true) == sortperm(collect(pa); rev = true)
            pm = PooledArray(rand([missing, "b", "a", "c"], n))
            @test psortperm(pm) == sortperm(collect(pm))
        end
    end

    @testset "ChapelSortAlg and psortperm!" begin
        x = rand(Int, 100_000)
        @test sort(x; alg = ChapelSortAlg()) == sort(x)
        @test sortperm(x; alg = ChapelSortAlg(), rev = true) == sortperm(x; rev = true)
        @test sortperm(x; alg = ChapelSortAlg(), lt = >) == sortperm(x; lt = >)  # falls back
        a, b = rand(1:3, 200_000), PooledArray(rand(["x", "y", "z"], 200_000))
        idx = collect(200_000:-2:1)
        ref = idx[sortperm(collect(zip(a[idx], b[idx])))]
        @test psortperm!(copy(idx), (a, b)) == ref
        @test psortperm!(copy(idx), [a, b]; rev = [false, false]) == ref
    end
end
