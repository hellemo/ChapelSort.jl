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

"""
    ChapelSort

Experimental parallel sorting modelled on the Chapel standard library sort:
a two-array (out-of-place) MSD radix sort with parallel bucketization of
large ranges and parallel serial-radix sorting of small buckets.
"""
module ChapelSort

export psort!, psort, psortperm, psortperm!, ChapelSortAlg

# Extension points and helpers used by other packages (e.g. ParallelDataFrameOps).
@static if VERSION >= v"1.11"
    eval(Expr(:public, :sortkeys, :sortable, :orderspec, :radixkey, :isradixable,
              :isbytestring, :strbucket))
end

include("keys.jl")
include("strings.jl")
include("serial_msd.jl")
include("two_array.jl")
include("psort.jl")
include("sortperm.jl")

end
