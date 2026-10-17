# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Derived from the design of Chapel's standard library sort,
# modules/standard/Sort.chpl (string key parts)
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

# Byte-wise MSD radix keys for strings and symbols.
#
# Strings compare as `memcmp` followed by length, i.e. lexicographically by
# bytes with "end of string" smaller than any byte. Level l looks at byte l+1:
# bucket 1 holds strings that have ended (all identical within a bucket, since
# they share the l-byte prefix), buckets 2:257 hold byte values 0x00:0xff.

"""
    isbytestring(T::Type) -> Bool

Whether values of type `T` are sorted by [`psort!`](@ref) with the byte-wise
string radix sort. Extend together with `ChapelSort.strbucket`.
"""
isbytestring(::Type) = false
isbytestring(::Type{<:Union{String,SubString{String},Symbol}}) = true

@inline function strbucket(s::Union{String,SubString{String}}, level::Int)
    return level < ncodeunits(s) ? Int(@inbounds codeunit(s, level + 1)) + 2 : 1
end
@inline strbucket(s::AbstractString, level::Int) =
    level < ncodeunits(s) ? Int(codeunit(s, level + 1)::UInt8) + 2 : 1

# Symbols are NUL-terminated and compared with strcmp. A symbol in a bucket at
# `level` shares a NUL-free prefix of `level` bytes with its peers, so reading
# byte `level` is always within the name or its terminator.
@inline function strbucket(s::Symbol, level::Int)
    c = unsafe_load(Base.unsafe_convert(Ptr{UInt8}, s) + level)
    return c == 0x00 ? 1 : Int(c) + 2
end

"""Byte-wise string keys of `by(x)`; `R` selects reverse order."""
struct StrKey{R,F} <: KeySpec
    by::F
end
StrKey{R}(by::F) where {R,F} = StrKey{R,F}(by)

nbuckets(::Type{<:StrKey}) = 257
maxlevel(::StrKey) = DONE
@inline bucket(ks::StrKey{R}, x, level::Int) where {R} =
    (b = strbucket(ks.by(x), level); R ? 258 - b : b)
@inline recurse(::StrKey{R}, b::Int) where {R} = b != (R ? 257 : 1)
@inline keylt(ks::StrKey{false}, a, b) = isless(ks.by(a), ks.by(b))
@inline keylt(ks::StrKey{true}, a, b) = isless(ks.by(b), ks.by(a))
