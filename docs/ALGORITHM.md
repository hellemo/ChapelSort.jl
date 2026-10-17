# The ChapelSort algorithm

This document describes the sorting algorithms in ChapelSort.jl precisely enough
to reimplement them without reading any code. They are a Julia reimplementation
of the design of the Chapel standard library's parallel radix sort, presented by
Ferguson [14] and benchmarked in [15]. A few changes and additions are listed in
[§10](#10-deviations-from-chapel-and-the-literature). The design is described here
in prose, notation and parameter values only; this document contains no Chapel
source code. Chapel is © Hewlett Packard Enterprise Development LP and Cray Inc.,
licensed under Apache-2.0; see `NOTICE`.

Contents

1. [Overview and notation](#1-overview-and-notation)
2. [Order-preserving keys](#2-order-preserving-keys)
3. [The parallel two-array partition step](#3-the-parallel-two-array-partition-step)
4. [Skip-ahead over shared digits](#4-skip-ahead-over-shared-digits)
5. [Big tasks, small tasks and scheduling](#5-big-tasks-small-tasks-and-scheduling)
6. [The serial two-array MSD sort](#6-the-serial-two-array-msd-sort)
7. [Floating-point keys](#7-floating-point-keys)
8. [Stable sortperm, multi-column and pooled keys](#8-stable-sortperm-multi-column-and-pooled-keys)
9. [Complexity](#9-complexity)
10. [Deviations from Chapel and the literature](#10-deviations-from-chapel-and-the-literature)
11. [Measured performance](#11-measured-performance)
12. [References](#12-references)

---

## 1. Overview and notation

The sort is a **most-significant-digit-first (MSD) radix sort** with 8-bit digits
that uses one scratch array of the same length as the input. MSD radix sorting
and its engineering are classical [1, 2]. The novelty of the Chapel design [14] is
how it combines three things on a shared-memory multicore:

- a **parallel partition step** for large ranges, which moves data between the two
  arrays without locks or atomics (§3), as in the per-processor histogram plus
  prefix-sum scheme of parallel radix sorts [3, 4, 5, 6];
- **independent serial sorts** for the many small buckets the partition steps
  produce, run in parallel (§5, §6);
- **skip-ahead** over digits that all keys share (§4).

Notation:

| Symbol | Meaning |
|---|---|
| $A$, $S$ | the input array and a scratch array of the same length and index range |
| $n$ | number of elements |
| $p$ | number of tasks (threads) used; default: the number of Julia threads |
| $k(x)$ | the **key** of element $x$: an unsigned integer of $w$ bits (§2), or a byte string |
| $L$ | number of digit levels: $w/8$ for fixed-width keys, unbounded for strings |
| $d_\ell(x)$ | the digit of $x$ at **level** $\ell = 0, 1, \dots$ (level 0 is most significant) |
| $B$ | number of buckets per level: 256 (fixed-width keys) or 257 (strings) |
| $[l, h]$ | a contiguous index range of the arrays, $m = h - l + 1$ elements |

For a fixed-width key $k(x) \in \{0, \dots, 2^w - 1\}$ the digit at level $\ell$ is
byte $\ell$ counted from the top. Buckets are numbered from $1$:

$$
d_\ell(x) \;=\; 1 + \left\lfloor \frac{k(x)}{2^{\,w - 8(\ell+1)}} \right\rfloor \bmod 256,
\qquad 0 \le \ell < L = w/8 .
$$

**Invariant of MSD radix sorting.** When a range $[l,h]$ is processed at level
$\ell$, all its keys agree on digits $0,\dots,\ell-1$. Distributing the range into
buckets by $d_\ell$, in increasing bucket order, puts it in sorted order with
respect to the first $\ell+1$ digits. Each bucket is then sorted recursively at
level $\ell+1$. Recursion stops at ranges of one element, ranges whose keys are
exhausted ($\ell = L$, or an "ended" string bucket), or small ranges handed to a
comparison base case.

All distribution steps are **stable**: elements with equal digits keep their
relative order. The base case (insertion sort) is stable too, so the whole sort
is stable. That stability is what makes `sortperm` and multi-column sorting (§8)
correct.

---

## 2. Order-preserving keys

Every element type is mapped to an unsigned integer key $k$ so that integer
comparison of keys reproduces Julia's `isless` exactly:
$\;a \prec b \iff k(a) < k(b)$ for unequal keys.

| Element type | Key $k(x)$ |
|---|---|
| Unsigned integers | $x$ itself |
| Signed $w$-bit integers (two's complement) | $\mathrm{bits}(x) \oplus 2^{w-1}$, i.e. flip the sign bit: $-2^{w-1} \mapsto 0$, $2^{w-1}-1 \mapsto 2^w - 1$ |
| `Bool`, `Char` | the 8-bit value; the 32-bit code value (Julia orders `Char` by it) |
| IEEE floats (`Float32`, `Float64`) | see below |
| Short InlineStrings (`String1/3/7/15`) | the raw 16/32/64/128-bit value |
| Reverse order (`rev = true`) | $\lnot k(x)$, the bitwise complement, which reverses integer order |
| `by = f` | the key of $f(x)$ |

**Floats** [13]. Let $u = \mathrm{bits}(x)$ be the $w$-bit pattern and $s$ the
sign bit. Then

$$
k(x) = \begin{cases} \lnot u & \text{if } s = 1 \quad (\text{negative: flip all bits}) \\
u \oplus 2^{w-1} & \text{if } s = 0 \quad (\text{non-negative: flip the sign bit}). \end{cases}
$$

This orders $-\infty < \dots < -0.0 < +0.0 < \dots < +\infty$, which agrees with
`isless`, including $-0.0 \prec +0.0$. NaNs need separate care. `isless` places
every NaN, of either sign and any payload, after $+\infty$. Under the map above,
negative-signed NaNs would land first. ChapelSort therefore either maps all NaNs
to $2^w - 1$ (per-element keys), or moves NaNs out before sorting (the in-place
transform of §7). Reverse order places NaNs first, matching Julia's `sort(...; rev = true)`.

**InlineStrings.** A fixed-width inline string of capacity $c-1$ bytes is stored as
one $8c$-bit integer. The characters sit most-significant byte first, unused bytes
are zero, and the length is in the least significant byte. Comparison between two
such values is plain unsigned integer comparison, so the raw bits are already an
order-preserving key. Zero padding plus the trailing length byte order a string
before its extensions, so `"a" ≺ "a\0"`.

**Strings and symbols (variable length).** These are compared lexicographically
by bytes, and a string sorts before every extension of itself. That is `memcmp`
followed by length for `String`, and `strcmp` up to the terminating NUL for
`Symbol`. The radix key is the byte sequence, with $B = 257$ buckets per level:

$$
d_\ell(s) = \begin{cases} 1 & \text{if } \ell \ge |s| \quad(\text{the string has ended}) \\
2 + s_{\ell+1} & \text{otherwise, where } s_{\ell+1} \in \{0,\dots,255\} \text{ is byte } \ell+1.\end{cases}
$$

Bucket 1, "ended", sorts before every byte value, which gives the
prefix-before-extension rule. All strings in an ended bucket at level $\ell$ share
the same $\ell$ bytes and have length $\ell$, so they are **equal**. That bucket
is never recursed into, and this is how the recursion terminates for strings.
Embedded zero bytes and invalid UTF-8 need no special handling, because only bytes
are compared. For `Symbol`s, "ended" means the NUL terminator. Reading byte $\ell$
is always safe: every symbol in a range at level $\ell$ shares a NUL-free prefix
of length $\ell$. In reverse order, bucket $b$ becomes $B + 1 - b$, so "ended" is
bucket 257 and is again not recursed into.

The base case (insertion sort) compares complete keys: $k(a) < k(b)$ for
fixed-width keys, and full lexicographic comparison for strings.

---

## 3. The parallel two-array partition step

The parallel step distributes a large range $[l,h]$ by one digit level $\ell$ from
a source array $X$ into a destination array $Y$ (one of $A, S$ each), using $p$
tasks. It runs in three phases. Each phase is a parallel loop over $p$
**contiguous blocks** of the range: block $t \in \{1,\dots,p\}$ is

$$
\mathrm{blk}_t = \left[\, l + \left\lfloor \tfrac{(t-1)m}{p} \right\rfloor,\;
l + \left\lfloor \tfrac{t\,m}{p} \right\rfloor - 1 \,\right].
$$

**Phase 1: count.** Task $t$ builds a private histogram of its block:

$$
c_{b,t} = \bigl|\{\, i \in \mathrm{blk}_t : d_\ell(X_i) = b \,\}\bigr| ,\qquad b = 1..B .
$$

The bucket totals are $C_b = \sum_t c_{b,t}$.

**Phase 2: bucket-major exclusive prefix sum.** Order the pairs $(b,t)$
lexicographically, by bucket first and task second, and compute

$$
o_{b,t} \;=\; l + \sum_{(b',t') < (b,t)} c_{b',t'}
\;=\; l + \sum_{b' < b} C_{b'} + \sum_{t' < t} c_{b,t'} .
$$

$o_{b,t}$ is where task $t$ writes its first element of bucket $b$. This sum has
only $Bp$ terms, so it is computed serially.

**Phase 3: scatter.** Task $t$ scans its block in index order. For each element
$x$ with digit $b$ it writes $Y_{o_{b,t}} \leftarrow x$ and increments $o_{b,t}$.

**Correctness and freedom from races.** Task $t$ writes exactly the
$c_{b,t}$ positions $[\,o_{b,t},\; o_{b,t} + c_{b,t} - 1\,]$ for bucket $b$. By
construction of the prefix sum these intervals tile $[l,h]$ without overlap.
So every element is written exactly once, no two tasks write the same location,
and no synchronisation is needed apart from the barriers between phases.

**Stability.** Within bucket $b$, task $t$'s elements land after those of every
task $t' < t$, whose blocks come earlier in the source. Inside one task they keep
scan order. So equal digits keep their source order.

**Two arrays.** After the step, bucket $b$ occupies
$[\,l + \sum_{b'<b} C_{b'},\; l + \sum_{b' \le b} C_{b'} - 1\,]$ of $Y$. Its
sub-sorts read from $Y$ and write into $X$: the roles **alternate** at every level.
Each task record therefore carries a flag saying which array currently holds its
data. The final result must end up in $A$, so a range that finishes in $S$ gets a
**copy-back** (§5).

The whole step is the parallel analogue of one pass of a counting sort [1], and
the same scheme as the parallel radix sorts of [3, 5, 6], applied to one MSD level
of one range.

---

## 4. Skip-ahead over shared digits

Before scattering, the step checks whether one bucket received all $m$ elements,
i.e. $\max_b C_b = m$. If so, the scatter would only copy the range unchanged, so
it is skipped:

- If that single bucket must not be recursed into (the "ended" string bucket), or
  the last level has been reached, all keys in the range are **equal**. The range
  is done and its data stays where it is.
- Otherwise move to level $\ell+1$ and **count again**, with no data movement.

Inputs whose keys share long prefixes then cost only one counting pass per
shared digit. Examples are small integers stored in 64 bits, timestamps from a
narrow window, and strings like `"id0000012345"`. The serial sort of §6 applies
the same rule.

---

## 5. Big tasks, small tasks and scheduling

A **task record** is $(l, h, \ell, \text{inA})$: a range, the level at which to
continue, and which array holds its data. A special level value, written
$\ell = \top$, marks ranges that only need copying back to $A$.

With $n$ the full input length, a bucket is **big** when its size exceeds

$$
T \;=\; \max\!\left(\left\lceil \tfrac{n}{p} \right\rceil,\; 4096\,p\right),
$$

and **small** otherwise. $T$ is fixed once from the full input length. A big
range is worth a parallel partition step. A small one is cheaper to sort serially
on one task while other tasks sort other small ranges.

**Driver.**

1. If $n < 50\,000$, or $p = 1$, sort serially (§6) and stop.
2. Put $(1, n, 0, \text{inA}=\text{true})$ on the *big* stack if $n > T$,
   otherwise on the *small* list.
3. While the big stack is non-empty, pop a record and run the parallel step (§3,
   with skip-ahead §4) from its current array into the other one.
   - If it found all keys equal, add a copy-back record if the data is in $S$.
   - Otherwise, for every non-empty bucket $b$ (size $C_b$, at the next level
     $\ell' = \ell + 1$, now in the other array):
     - if $C_b = 1$, or $b$ is a non-recursing bucket, or $\ell' = L$: the bucket
       is final. Add a copy-back record if it lies in $S$;
     - else if $C_b > T$: push it on the big stack;
     - else: append it to the small list.
4. **Small phase.** Sort the small list by size, largest first. Start
   $\min(p, \#\text{small})$ worker tasks. Each repeatedly takes the next unclaimed
   record from a shared atomic counter, then:
   - copy-back records: copy the range from $S$ to $A$;
   - others: run the serial sort (§6) at level $\ell$ from the array the record
     names, which leaves the result in $A$.

Largest-first dynamic assignment is the classic LPT (longest processing time)
heuristic. It keeps one unlucky large bucket from finishing last. The big
partition steps run one at a time, each using all $p$ tasks.

For $n = 2^{27}$ uniformly random 64-bit keys and $p = 8$: $T = 2^{24}$, the first
partition step produces 256 buckets of about $2^{19}$ elements, all small, and the
small phase sorts them in parallel.

---

## 6. The serial two-array MSD sort

The serial sort handles one range on one task. It is the same algorithm as the
parallel step without the per-task decomposition ($p = 1$). The other array's
range $[l,h]$ serves as scratch.

```
serial_sort(l, h, ℓ, data in X, other array Y):
    m = h − l + 1
    if m ≤ 48 or ℓ = L (or ℓ = ⊤):
        if data is in S: copy X[l..h] → A[l..h]
        if ℓ < L and m > 1: insertion-sort A[l..h] by full keys
        return
    repeat:                                   # skip-ahead (§4)
        count C_b = |{i ∈ [l,h] : d_ℓ(X_i) = b}|
        if max_b C_b < m: break
        if that bucket does not recurse or ℓ + 1 = L:
            if data is in S: copy X[l..h] → A[l..h]
            return
        ℓ = ℓ + 1
    o_b = l + Σ_{b'<b} C_{b'};   scatter X[l..h] → Y stably using o_b
    for each non-empty bucket b with range [s, e] in Y:
        serial_sort(s, e, recurse(b) ? ℓ + 1 : ⊤, data in Y, other array X)
```

The histogram for level $\ell$ must survive while that level's children are sorted.
One count vector per level is therefore kept and reused: $L$ vectors of $B$
counters for fixed-width keys, and a list that grows on demand for strings.
Nothing is allocated per call.

**Base case.** Ranges of at most 48 elements use straight insertion sort, shifting
an element left while it is strictly smaller than its predecessor. Using strict
comparison keeps the sort stable. The cutoff is a tuning parameter: below about
50 elements, the $O(B)$ cost of clearing and scanning a histogram outweighs the
$O(m^2)$ comparisons.

---

## 7. Floating-point keys

Computing the float key (§2) needs a sign test and a NaN test. Doing that again in
every counting pass, scatter pass and insertion comparison measurably slowed down
sorting `Float64`. When sorting floats by value (no `by`), ChapelSort therefore
transforms the array **in place, once**:

1. **NaN partition.** Count the NaNs. If there are any, partition the array stably
   through the scratch array: non-NaNs first and NaNs last, or NaNs first for
   reverse order. Only the non-NaN part is sorted further.
2. **Forward map.** Reinterpret each element's bits $u$ as an unsigned integer and
   overwrite it with $k(u)$ from §2, complemented for reverse order. This is a
   parallel map over contiguous blocks.
3. Sort the now-integer array with the integer radix sort (§3–§6).
4. **Inverse map.** For each key $k$ (first complemented back in reverse order),
   restore $u = k \oplus 2^{w-1}$ if the top bit of $k$ is set, otherwise
   $u = \lnot k$.

Without NaNs, the map of §2 is a bijection on bit patterns, so the original bits
are restored exactly, including $-0.0$ and subnormals. The two extra parallel
passes are cheap next to the sort. Julia's `Base` sort uses the same idea for its
serial radix sort of floats.

---

## 8. Stable sortperm, multi-column and pooled keys

**Records.** `sortperm` sorts **records** $(k(x_i), i)$ by their key alone. Since
the sort is stable, records with equal keys keep increasing $i$, which is exactly
the stable permutation `sortperm` must return. The index is stored as a 32-bit
integer when $n \le 2^{32} - 1$.

**Packing.** When the key has at most 32 bits and $n \le 2^{32}$, each record
becomes one 64-bit word,

$$
r_i = k(x_i)\cdot 2^{32} + (i - 1),
$$

and the radix sort treats only the high 32 bits as the key (levels $0..3$ of $r$).
The low half is never inspected, so a long run of equal keys costs no extra
passes, and stability still gives increasing $i$ within equal keys.

**Multiple columns** $(c_1, \dots, c_q)$ in lexicographic order, each with its own
direction. This uses the classical **least-significant-column-first** method,
the column-level analogue of LSD radix sorting [1]. Start from the identity
permutation $\pi$, or a given subset of row indices. For $j = q, q-1, \dots, 1$:

1. gather the keys $c_j[\pi_1], \dots, c_j[\pi_m]$;
2. compute their stable sortperm $\sigma$ (in direction $j$);
3. set $\pi \leftarrow \pi \circ \sigma$, i.e. $\pi_i \leftarrow \pi_{\sigma_i}$.

By induction, after processing columns $j..q$, $\pi$ orders the rows
lexicographically by $(c_j, \dots, c_q)$. Pass $j$ orders by $c_j$, and stability
keeps the order from columns $j+1..q$ among rows with equal $c_j$.

**Pooled (dictionary-encoded) columns.** A column stored as small integer
references into a pool of distinct values is sorted by **rank**:

1. sort the pool (it is small);
2. give every pool entry its rank in sorted order, with equal values sharing a
   rank and the direction applied here;
3. sort the rows by the integer rank of their reference, using the packed records
   above.

The string comparisons then happen only on the pool, never per row.

---

## 9. Complexity

Let $L_{\text{eff}} \le L$ be the number of levels that actually split data,
roughly $\lceil \log_{256} n \rceil$ for random keys.

- **Work.** Each level costs one counting and one scatter pass over the data it
  touches, plus $O(Bp)$ for the prefix sum. Total work is
  $O\bigl(n \cdot L_{\text{eff}} + (\#\text{ranges}) \cdot B\bigr)$, plus one
  counting pass per skipped shared digit. The base case adds $O(48\,n)$ in the
  worst case. For strings, $L$ is the length of the distinguishing prefix: the
  usual $O(D + n)$ string-sorting bound plus $B$ per range [2, 10].
- **Span.** A parallel partition step has span $O(m/p + Bp)$. The small phase
  has span $O\!\left(\frac{1}{p}\sum \text{work} + \max_{\text{small range}} \text{work}\right)$,
  which is bounded because small ranges have at most $T \approx n/p$ elements.
- **Memory.** One scratch array of $n$ elements, $p \cdot B$ counters for the
  parallel step, and $L \cdot B$ counters per worker task. Records for
  `sortperm` add $n$ records (8 or 16 bytes each) and their scratch.

Performance is bound by memory bandwidth. Each MSD level reads and writes the data
once. The parallel steps run at DRAM bandwidth, and the small phase works on
cache-sized ranges.

---

## 10. Deviations from Chapel and the literature

The reasons below were measured on the benchmark machine (8-core laptop, Julia
1.13; see §11).

| Aspect | Chapel standard library [14, 16] | ChapelSort | Reason |
|---|---|---|---|
| Stability | default `sort` is unstable | stable everywhere | needed for `sortperm`, multi-column sorts and `by`; costs nothing with an out-of-place base sort |
| Serial sort of small ranges | in-place MSD radix (American-flag style permutation [2]) | out-of-place two-array MSD (§6), using the free half of the scratch array | stable, and the plain scatter is faster than a cycle-following permutation here |
| Base case | shell sort below 16 elements | insertion sort at ≤ 48 elements | fewer histogram passes on small ranges; measured faster |
| Float keys | order-preserving key part computed per comparison/pass | in-place one-time key transform (§7) | about 30–40% faster for `Float64` |
| Variable-length keys | key parts with an end-of-key status | explicit 257th "ended" bucket that is never recursed into | same order; keeps the inner loops identical to the fixed-width case |
| Thresholds | serial below 50 000; small-range threshold from tasks × 4096 and $n/p$ | same values (§5) | — |
| Small-task scheduling | parallel loop over small ranges | largest-first, atomic-counter work queue | balances unequal buckets |
| Scatter buffers | none | none: software write-combining buffers [6, 7] were tried and were **slower** (1.5–2.1 vs 1.0–1.5 ns/element) | Julia's plain scatter is already TLB-friendly enough at 256 buckets on this machine |
| `sortperm` | — | packed key/index words for ≤ 32-bit keys (§8) | avoids extra passes on low index bits |
| Pooled columns | — | sort by pool rank (§8) | string comparisons only on the pool |

Julia-specific requirements, each of which cost 1.5× to 20× when violated:

- Key-extractor objects must **not** be subtypes of `Function`. Otherwise Julia
  may not specialize a method that only passes them on, and the recursion boxes
  values.
- Histograms are plain vectors, not views into a matrix. A view that stays live
  across a recursive call is heap-allocated.
- A variable captured by a parallel closure must not be reassigned (for example
  in two `if` branches), or Julia boxes it. The fix is a separate function (a
  "function barrier").

Not implemented, but discussed in the literature: in-place parallel variants
(IPS²Ra [7], Regions Sort [8], American flag [2]) that avoid the $n$-element
scratch array; sample-sort front ends for arbitrary comparison functions and
skewed keys [12]; caching several key bytes per string record to reduce pointer
chasing [10, 11].

---

## 11. Measured performance

Measured on an Intel Core Ultra 7 268V laptop (4 performance and 4 efficiency
cores, 8 threads), Julia 1.13.1, Chapel 2.10.0, sorting $2^{27}$ random `UInt64`
(1 GiB):

| | 1 task | 8 tasks |
|---|---:|---:|
| `Base.sort!` (serial) | 3.04 s | — |
| Chapel 2.10 `sort` (scratch allocated per call) | 2.41 s | 0.52 s |
| ChapelSort `psort!`, fresh scratch | 2.29 s | 0.45 s |
| ChapelSort `psort!`, reused scratch | 2.14 s | 0.38 s |

For $10^7$ values at 8 threads, against Base: `String` 9–16× (`sortperm` 13–25×),
`Symbol` 13×, `Float64` 5×, `Int64` 6× (`sortperm` 19–22×). Benchmark scripts and
raw results are in `bench/`, and charts and tables are in the [README](../README.md).

---

## 12. References

1. D. E. Knuth. *The Art of Computer Programming, Vol. 3: Sorting and Searching*,
   2nd ed. Addison-Wesley, 1998. ISBN 0-201-89685-0. §5.2 (sorting by counting,
   incl. distribution counting) and §5.2.5 (sorting by distribution, i.e. radix
   sorting).
2. P. M. McIlroy, K. Bostic, M. D. McIlroy. "Engineering Radix Sort."
   *Computing Systems* 6(1):5–27, 1993.
   <https://www.usenix.org/legacy/publications/compsystems/1993/win_mcilroy.pdf>.
   MSD byte-wise radix sorts: a stable two-array sort and the in-place
   "American flag" sort.
3. M. Zagha, G. E. Blelloch. "Radix Sort for Vector Multiprocessors."
   *Proc. Supercomputing '91*, pp. 712–721, 1991. doi:10.1145/125826.126164.
   Parallel (vectorized) radix sort built on parallel histograms and scans.
4. G. E. Blelloch. "Prefix Sums and Their Applications." Tech. Rep.
   CMU-CS-90-190, Carnegie Mellon University, 1990.
   <https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html>. Also in
   J. H. Reif (ed.), *Synthesis of Parallel Algorithms*, Morgan Kaufmann, 1993.
5. N. Satish, M. Harris, M. Garland. "Designing Efficient Sorting Algorithms for
   Manycore GPUs." *Proc. IEEE IPDPS 2009*, pp. 1–10. doi:10.1109/IPDPS.2009.5161005.
   Per-block digit histograms followed by a prefix sum over the
   (blocks × buckets) table to obtain global scatter offsets.
6. O. Polychroniou, K. A. Ross. "A Comprehensive Study of Main-Memory
   Partitioning and its Application to Large-Scale Comparison- and Radix-Sort."
   *Proc. ACM SIGMOD 2014*, pp. 755–766. doi:10.1145/2588555.2610522. Parallel
   partitioning from prefix sums of per-thread histograms, and cache-line
   write-combining buffers.
7. M. Axtmann, S. Witt, D. Ferizovic, P. Sanders. "Engineering In-place
   (Shared-memory) Sorting Algorithms." *ACM Trans. Parallel Comput.* 9(1),
   Article 2, 2022. doi:10.1145/3505286. IPS⁴o and IPS²Ra. Earlier: "In-Place
   Parallel Super Scalar Samplesort (IPSSSSo)", *ESA 2017*, LIPIcs 87, 9:1–9:14.
   doi:10.4230/LIPIcs.ESA.2017.9.
8. O. Obeya, E. Kahssay, E. Fan, J. Shun. "Theoretically-Efficient and Practical
   Parallel In-Place Radix Sorting." *Proc. SPAA 2019*, pp. 213–224.
   doi:10.1145/3323165.3323198.
9. M. Kokot, S. Deorowicz, M. Długosz. "Even Faster Sorting of (Not Only)
   Integers." In *Man-Machine Interactions 5 (ICMMI 2017)*, AISC 659,
   pp. 481–491, Springer, 2017. doi:10.1007/978-3-319-67792-7_47
   (arXiv:1703.00687). RADULS2, a parallel MSD radix sort for multicore CPUs.
10. J. Kärkkäinen, T. Rantala. "Engineering Radix Sort for Strings."
    *SPIRE 2008*, LNCS 5280, pp. 3–14. doi:10.1007/978-3-540-89097-3_3.
11. J. L. Bentley, R. Sedgewick. "Fast Algorithms for Sorting and Searching
    Strings." *Proc. 8th ACM-SIAM SODA*, pp. 360–369, 1997.
    <https://www.cs.tufts.edu/~nr/cs257/archive/bob-sedgewick/fast-strings.pdf>.
    Multikey quicksort.
12. P. Sanders, S. Winkel. "Super Scalar Sample Sort." *ESA 2004*, LNCS 3221,
    pp. 784–796. doi:10.1007/978-3-540-30140-0_69.
13. M. Herf. "Radix Tricks." Web article, December 2001.
    <http://stereopsis.com/radix.html>. The order-preserving float-to-integer
    key transform.
14. M. Ferguson. "Towards Radix Sorting in the Chapel Standard Library."
    CHIUW 2019 (slides). <https://chapel-lang.org/CHIUW/2019/Ferguson.pdf>.
    MSD counting radix sorts in Chapel: a recursive sort with serial in-place
    bucketize, and a two-array sort with parallel count, scan and bucketize into
    scratch.
15. M. Ferguson. "Comparing Standard Library Sorts: The Impact of Parallelism."
    Chapel Language Blog, January 30, 2024.
    <https://chapel-lang.org/blog/posts/std-sort-performance/>.
16. Chapel `Sort` module documentation.
    <https://chapel-lang.org/docs/modules/standard/Sort.html>.
17. M. Ferguson. "Distributed-Memory Sorting in the Chapel Standard Library."
    ChapelCon '25, 2025 (slides).
    <https://chapel-lang.org/ChapelCon/2025/chapelcon-2025-sort.pdf>. Not
    implemented here; relevant for a distributed extension.
