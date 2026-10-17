// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lars Hellemo
//
// Derived from the benchmark in the Chapel blog post "Comparing Standard
// Library Sorts: The Impact of Parallelism" (M. Ferguson, 2024,
// https://chapel-lang.org/blog/posts/std-sort-performance/).
// Chapel: Copyright 2020-2026 Hewlett Packard Enterprise Development LP;
// Copyright 2004-2019 Cray Inc.; Apache License 2.0.
// Modified: repetitions, task counts and CSV output added; see NOTICE.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Port of the benchmark from https://chapel-lang.org/blog/posts/std-sort-performance/
// Build: chpl --fast sortbench.chpl    Run: ./sortbench --log2n=27 --reps=5
use Random, Time, Sort;

config const log2n = 27;
config const reps = 5;
config const csv = false;  // print "method,ntasks,n,seconds" row instead

proc main() {
  const n = 1 << log2n;
  var X: [0..<n] uint;
  fillRandom(X);
  var A: [0..<n] uint;

  A = X; sort(A);  // warm up
  if !isSorted(A) then halt("not sorted");

  var total = 0.0;
  for 1..reps {
    A = X;
    var t: stopwatch;
    t.start();
    sort(A);
    t.stop();
    total += t.elapsed();
  }
  const mean = total / reps;
  if csv then
    writeln("Chapel sort,", here.maxTaskPar, ",", n, ",", mean);
  else
    writeln("Chapel sort 2^", log2n, " uint(64), maxTaskPar=", here.maxTaskPar,
            ": ", mean, " s  ", n / mean / 1e6, " M elem/s");
}
