#!/usr/bin/env bash
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

# Build and run the Chapel benchmark for several task counts; writes ../results/chapel.csv
# Usage: bench/chapel/run.sh [log2n] [reps] [tasks...]
set -euo pipefail
cd "$(dirname "$0")"
log2n=${1:-27}; reps=${2:-5}; shift 2 || true
tasks=${*:-1 2 4 8}
[ sortbench -nt sortbench.chpl ] || chpl --fast sortbench.chpl -o sortbench
mkdir -p ../results
out=../results/chapel.csv
echo "method,ntasks,n,seconds" > "$out"
for k in $tasks; do
  CHPL_RT_NUM_THREADS_PER_LOCALE=$k ./sortbench --log2n="$log2n" --reps="$reps" --csv=true \
    --dataParTasksPerLocale="$k" | tee -a "$out"
done
