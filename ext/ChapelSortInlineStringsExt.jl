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

module ChapelSortInlineStringsExt

# InlineStrings store the bytes most-significant first and the length in the
# lowest byte, and compare as unsigned integers of the same width, so the raw
# bits are the radix key. Wider types use the byte-wise string path.

using ChapelSort, InlineStrings

for (T, U) in ((String1, UInt16), (String3, UInt32), (String7, UInt64), (String15, UInt128))
    @eval ChapelSort.isradixable(::Type{$T}) = true
    @eval @inline ChapelSort.radixkey(x::$T) = reinterpret($U, x)
end
ChapelSort.isbytestring(::Type{<:InlineString}) = true

end
