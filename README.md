# ChapelSort.jl

Experimental parallel sort for Julia, modelled on the Chapel standard library
sort described in
[“Standard Library Sorting Performance”](https://chapel-lang.org/blog/posts/std-sort-performance/)
(2^27 random `UInt64` in 0.25 s on a 16-core Ryzen 7950X).

```julia
using ChapelSort
psort!(v)                          # in place, uses Threads.nthreads() tasks
psort(v; rev = true, by = first)   # copy; `by` must return a radix-sortable value
psort!(v; ntasks = 4, scratch = s) # reuse a scratch vector (same axes/eltype)
psortperm(v)                       # stable, like sortperm
psortperm((a, b, c); rev = [true, false, false])   # lexicographic over columns
psortperm!(idx, (a, b))            # stably reorder given row indices
sort(v; alg = ChapelSortAlg())     # as a Base.Sort.Algorithm
```

DataFrames joins, grouping and `sort!(df, cols; alg = ChapelSortAlg())` live in the
companion package [ParallelDataFrameOps.jl](../ParallelDataFrameOps.jl), which is
MIT licensed and built on this one.

**How it works:** [docs/ALGORITHM.md](docs/ALGORITHM.md) describes the algorithms
in enough detail to reimplement them, with references and the deviations from
Chapel's design.

Supported keys, all ordered exactly as `isless` and sorted **stably**:
- `Base.BitInteger`, `Bool`, `Char`, `Float32`, `Float64` (`-0.0 < 0.0`, NaNs last).
- `String`, `SubString{String}`, `Symbol`: byte-wise MSD radix with an "end of
  string" bucket, so embedded `\0`, non-ASCII and invalid UTF-8 are all fine.
- InlineStrings (extension): `String1/3/7/15` are radix sorted by their raw bits,
  since their `isless` is unsigned integer comparison. Wider types go byte-wise.
- `PooledArray` (extension, for `psortperm` and multi-column sorts): sort the
  small pool, then radix sort the integer ranks.

Other types, and custom `lt`, fall back to Base. To add a type, extend
`ChapelSort.radixkey` and `ChapelSort.isradixable`, or `ChapelSort.isbytestring`
and `ChapelSort.strbucket`.

## Algorithm (Chapel's `TwoArrayPartitioning` + `MSBRadixSort`)

In short (details, notation and references in [docs/ALGORITHM.md](docs/ALGORITHM.md)):

The algorithm is an out-of-place MSD radix sort with 8-bit digits and one scratch array:

1. **Parallel bucketize** of a large range, by one digit from `A` into `Scratch`:
   - Each of `ntasks` contiguous blocks counts its digits.
   - A bin-major prefix sum over the `256 × ntasks` counts gives every (bin, task)
     pair a private output region.
   - Each task scatters its block with no locks or atomics.
2. **Skip-ahead**: if all keys share the digit, try the next one without moving data.
   For fixed-width keys a sampled min/max pre-pass goes further. Fewer than 2^16
   distinct key values are sorted by one parallel counting sort; otherwise the sort
   starts at the first byte where min and max differ. On 2^26 keys in 0:1000 this
   takes 0.055 s, against 0.089 s for Base.
3. Buckets still larger than `max(n/ntasks, 4096·ntasks)` are bucketized again in
   parallel, in the opposite direction.
4. All other buckets become **small tasks**. They are processed largest first by a
   pool of tasks that pull from an atomic counter. Each one copies back if needed,
   then runs a serial two-array MSD radix sort, with insertion sort for ≤ 48
   elements.

Inputs below 50 000 elements (or `ntasks = 1`) use the serial sort directly.

Why chunk-sort + merge doesn't keep up:
- The final merges are bandwidth bound and poorly parallel.
- Each element is moved `log2(ntasks)` extra times.
- The chunk sorts compare instead of radix-sorting.

## Results

2^27 random `UInt64` (1 GiB), mean of 5 runs, on an Intel Core Ultra 7 268V laptop
(4 performance and 4 low-power efficiency cores) with Julia 1.13.2 and Chapel 2.10.0.
Chapel's `sort` allocates its scratch array on every call, so the fair comparison is
`psort!` with a *fresh* scratch array. The Chapel numbers come from an earlier session
on the same machine (Julia 1.13.1).

[ParallelRadixSort.jl (fork)](https://github.com/hellemo/ParallelRadixSort.jl) is a
fork of [tkf/ParallelRadixSort.jl](https://github.com/tkf/ParallelRadixSort.jl), a
parallel MSD radix sort for Julia from 2021. The fork makes it load on current Julia
and fixes thread safety, NaN, reverse-order and string bugs. Its default
`ParallelMSDRadixSort` is used here.

![Throughput vs number of tasks for psort! and Chapel sort](docs/scaling.svg)

![Throughput of all methods at 8 threads](docs/methods.svg)

| Method (8 threads)                | Time (s) | M elem/s |
|-----------------------------------|---------:|---------:|
| `Base.sort!` (serial)             |    3.147 |     42.7 |
| chunk sort + parallel merge       |    1.642 |     81.7 |
| `AcceleratedKernels.sort!`        |    1.292 |    103.9 |
| ParallelRadixSort.jl (fork)       |    0.794 |    169.1 |
| Chapel 2.10 `sort`                |    0.520 |    258.2 |
| `psort!`, fresh scratch           |    0.475 |    282.6 |
| `psort!`, reused scratch          |    0.394 |    340.8 |

| Tasks | Chapel `sort` (s) | `psort!` fresh (s) | `psort!` reused (s) |
|------:|------------------:|-------------------:|--------------------:|
| 1     | 2.409             | 2.502              | 2.242               |
| 2     | 1.268             | 1.166              | 1.111               |
| 4     | 0.692             | 0.724              | 0.595               |
| 8     | 0.520             | 0.475              | 0.394               |

Run-to-run noise on this laptop is ±10–15%.

Floats are converted in place to unsigned keys, sorted as integers, and converted
back. This costs two cheap parallel passes. Computing the float key on every pass
instead made `Float64` 30–40% slower.

### Strings, symbols and sortperm

From `bench/strings.jl`: 10^7 values, 8 threads, best of 3, against Base `sort!`/`sortperm`.

![Speedup of psort! and psortperm over Base for strings, symbols and numbers](docs/strings.svg)

| Values (10^7)                      | `sort!` | `psort!` | × | ParallelRadixSort.jl (fork) | × | `sortperm` | `psortperm` | × |
|------------------------------------|--------:|---------:|--:|---------------:|--:|-----------:|------------:|--:|
| `String`, random, length 8–20      | 4.98 s  | 0.305 s | 16.3 | 0.341 s | 14.6 | 11.24 s | 0.457 s | 24.6 |
| `String`, h2o id3 (`id%010d`)      | 2.41 s  | 0.279 s |  8.6 | 0.429 s |  5.6 |  5.65 s | 0.390 s | 14.5 |
| `String15`, h2o id3                | 0.251 s | 0.076 s |  3.3 | 0.253 s |  1.0 |  1.65 s | 0.160 s | 10.3 |
| `String7`, 100 levels              | 0.067 s | 0.036 s |  1.9 | 0.083 s |  0.8 |  1.33 s | 0.073 s | 18.4 |
| `Symbol`, 10^5 distinct            | 1.30 s  | 0.095 s | 13.7 | unsupported | — |  3.80 s | 0.225 s | 16.9 |
| `Int64`                            | 0.234 s | 0.040 s |  5.8 | 0.066 s |  3.6 |  1.49 s | 0.073 s | 20.3 |
| `Float64` (randn)                  | 0.255 s | 0.043 s |  5.9 | 0.062 s |  4.1 |  1.56 s | 0.078 s | 20.0 |
| `PooledArray{String}`, 10^5 levels | — | — | — | — | — | 0.151 s | 0.051 s | 2.9 |

- InlineStrings already ship a fast serial LSD radix sort for `sort!`. For
  `PooledArray`, PooledArrays already does sort the pool, then counting-sort, serially.
  Both are smaller wins.
- `psortperm` sorts (key, index) records with the same radix machinery. Keys of up
  to 32 bits are packed with the index into one `UInt64`.
- ParallelRadixSort.jl has no `sortperm` and no `Symbol` support. It reads InlineStrings
  byte by byte rather than as one integer key.

### Reproducing

```sh
bench/chapel/run.sh 27 5                              # needs `chpl`; writes bench/results/chapel.csv
julia -t 8 --project=bench bench/bench.jl 27 5        # writes bench/results/julia.csv
julia -t 8 --project=bench bench/strings.jl 1e7 3     # writes bench/results/strings.csv
julia --project=bench bench/plots.jl                  # AlgebraOfGraphics → docs/*.svg
```

The bench environment expects the ParallelRadixSort.jl fork in a sibling directory
(`../ParallelRadixSort.jl`, see `bench/Project.toml`). Sizes are chosen for a 16 GB
machine: 2^27 `UInt64` is 1 GiB, plus 1 GiB of scratch.

## Implementation notes / lessons

- The key extractors (`Fwd`, `Rev`) must **not** subtype `Function`. If they do,
  Julia does not specialize functions that only pass them along. That boxed values
  in the recursion and made the sort about 1.5× slower overall.
- Count arrays are indexed as `counts[bin, col]`, not through views. A view that stays
  alive across a recursive call is heap allocated.
- Write-combining scatter buffers (the IPS²Ra trick) were *slower* than a plain
  scatter on this machine, so they are not used.
- Allocating the scratch array costs about 0.09 s per GiB in page faults. Pass
  `scratch` to reuse it.
- A captured variable that is assigned more than once (for example in two `if`
  branches) is boxed by Julia's closure lowering. Inside the parallel `do` blocks
  this cost 10–20×, which happened twice (float keys, group indices). The fix is a
  function barrier.
- One core with an interface for keys (`nbuckets`, `maxlevel`, `bucket`,
  `recurse`, `keylt`) serves fixed-width keys (256 buckets) and strings (257
  buckets, unbounded depth). The fixed-width path compiles to the same code as before.

## Ideas not yet done

- A sample-sort bucketizer (splitters + branchless search) for arbitrary `lt`.
- Strings: cache 8 key bytes per record so each level doesn't chase a pointer per
  string.
- `missing` keys in `psort!`/`psortperm` (sorted last). Today they only work through
  `PooledArray` or the Base fallback.

- Precomputed keys for expensive `by` functions.
- Running big-bucket partitions concurrently instead of one at a time.

## License

ChapelSort.jl is licensed under the [Apache License 2.0](LICENSE). It is an
independent Julia reimplementation of the design of the
[Chapel](https://github.com/chapel-lang/chapel) standard library sort (Copyright
2020-2026 Hewlett Packard Enterprise Development LP, Copyright 2004-2019 Cray Inc.,
Apache-2.0) and contains no Chapel source code. [NOTICE](NOTICE) lists the files
that follow Chapel's design, each of which carries a notice of modification.
