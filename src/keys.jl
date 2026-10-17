# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Derived from the design of Chapel's standard library sort,
# modules/standard/Sort.chpl (defaultComparator key parts)
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

# Order-preserving maps from sortable values to unsigned integers ("radix keys").
#
# `radixkey(x)` must satisfy `isless(a, b) == (radixkey(a) < radixkey(b))` for all
# non-equal keys. Extend `radixkey` and `isradixable` to support new types.

"""
    isradixable(T::Type) -> Bool

Whether values of type `T` can be sorted by radix sort via [`radixkey`](@ref).
"""
isradixable(::Type) = false
isradixable(::Type{<:Union{Base.BitInteger,Bool,Char,Float32,Float64}}) = true

"""
    radixkey(x) -> Unsigned

Map `x` to an unsigned integer such that the integer order matches `isless`.
"""
radixkey(x::Unsigned) = x
radixkey(x::Bool) = UInt8(x)
radixkey(x::Char) = reinterpret(UInt32, x)
@inline function radixkey(x::T) where {T<:Base.BitSigned}
    U = unsigned(T)
    return reinterpret(U, x) ⊻ (one(U) << (8 * sizeof(T) - 1))
end
for (F, U) in ((Float32, UInt32), (Float64, UInt64))
    @eval @inline function radixkey(x::$F)
        u = reinterpret($U, x)
        k = u ⊻ ifelse(signbit(x), typemax($U), one($U) << (8 * sizeof($U) - 1))
        # `isless` places every NaN (regardless of sign) after +Inf.
        return ifelse(isnan(x), typemax($U), k)
    end
end

# Key extractors. Concrete callable structs keep everything type stable; they
# deliberately do not subtype `Function`, which would make Julia skip
# specializing functions that merely pass them on (boxing in the hot loops).
struct Fwd{F}
    by::F
end
@inline (k::Fwd)(x) = radixkey(k.by(x))

struct Rev{F}
    by::F
end
@inline (k::Rev)(x) = ~radixkey(k.by(x))

# Digit `level` (0 = most significant) of key `k`, as a 1-based bucket index.
@inline function digit(k::K, level::Int) where {K<:Unsigned}
    shift = 8 * (sizeof(K) - 1 - level)
    return Int((k >> shift) % UInt8) + 1
end

# ---------------------------------------------------------------------------
# Key specifications: what the radix sort needs to know about the elements.
#
#   nbuckets(ks)        number of buckets per level (a compile-time constant)
#   maxlevel(ks)        number of digit levels (typemax(Int) when unbounded)
#   bucket(ks, x, l)    bucket (1:nbuckets) of element x at level l (0 = first)
#   recurse(ks, b)      whether bucket b can still contain unequal keys
#   keylt(ks, a, b)     strict ordering, used by the insertion sort base case
# ---------------------------------------------------------------------------

abstract type KeySpec end

"""Fixed-width unsigned keys `kf(x)::K`, one byte per level."""
struct FixedKey{K<:Unsigned,F} <: KeySpec
    kf::F
end
FixedKey{K}(kf::F) where {K,F} = FixedKey{K,F}(kf)

nbuckets(::Type{<:FixedKey}) = 256
maxlevel(::FixedKey{K}) where {K} = sizeof(K)
@inline bucket(ks::FixedKey, x, level::Int) = digit(ks.kf(x), level)
@inline recurse(::FixedKey, b::Int) = true
@inline keylt(ks::FixedKey, a, b) = ks.kf(a) < ks.kf(b)

nbuckets(ks::KeySpec) = nbuckets(typeof(ks))

# Level marker for ranges that only need to be copied back, never sorted.
const DONE = typemax(Int)
