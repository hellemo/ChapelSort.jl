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

module ChapelSortPooledArraysExt

# Sort the (small) pool once, then radix sort the integer ranks of the refs.

using ChapelSort, PooledArrays
import ChapelSort: sortkeys, sortable, psortperm, foreach_block, nthreads_for

# rank[ref] = position of pool[ref] in sorted order (ties share a rank).
function poolranks(pa::PooledArray, rev::Bool)
    pool = pa.pool
    p = psortperm(pool; rev)
    rank = Vector{UInt32}(undef, length(pool) + 1)   # +1: ref 0 (#undef)
    rank[1] = 0
    r = UInt32(0)
    for (k, i) in enumerate(p)
        (k == 1 || !isequal(pool[p[k - 1]], pool[i])) && (r += UInt32(1))
        rank[i + 1] = r
    end
    return rank
end

sortable(::PooledArray, by) = by === identity

function sortkeys(pa::PooledArray, idx::AbstractVector{Int}, rev::Bool, nT::Int)
    rank = poolranks(pa, rev)
    refs = pa.refs
    keys = Vector{UInt32}(undef, length(idx))
    foreach_block(1, length(idx), nthreads_for(length(idx), nT)) do _, lo, hi
        @inbounds for i in lo:hi
            keys[i] = rank[Int(refs[idx[i]]) + 1]
        end
    end
    return keys, false
end

# All rows in order: no index vector to gather through.
function sortkeys(pa::PooledArray, ::Base.OneTo, rev::Bool, nT::Int)
    rank = poolranks(pa, rev)
    refs = pa.refs
    keys = Vector{UInt32}(undef, length(refs))
    foreach_block(1, length(refs), nthreads_for(length(refs), nT)) do _, lo, hi
        @inbounds for i in lo:hi
            keys[i] = rank[Int(refs[i]) + 1]
        end
    end
    return keys, false
end

function ChapelSort.psortperm(pa::PooledArray; by = identity, rev::Bool = false,
                              ntasks::Integer = Threads.nthreads())
    by === identity || return invoke(psortperm, Tuple{AbstractVector}, pa; by, rev, ntasks)
    keys, _ = sortkeys(pa, Base.OneTo(length(pa)), rev, Int(ntasks))
    return psortperm(keys; ntasks)
end

end
