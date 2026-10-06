# Thread Count Tests

Goal: decide whether `num_threads` should default to the number of physical
cores instead of Rayon's default (one thread per logical CPU). On the
development VM (8 physical cores, 16 vCPUs), 8 threads beat 16 by 2-18% at
depth 5 and 25-66% at depth 8. These tests check whether that holds on other
machines.

Run everything from the repository root. Expected total time: about 30-45
minutes, depending on the machine.

## 1. Get the branch onto the machine

The `feat/optimize` branch exists only as local commits on the development VM.
Either push it to a remote you control, or copy it as a bundle:

```sh
# On the development VM
git bundle create /tmp/forust-optimize.bundle main..feat/optimize

# On the test machine, inside an existing clone of the repository
git fetch /path/to/forust-optimize.bundle feat/optimize:feat/optimize
git checkout feat/optimize
```

## 2. Record the machine

```sh
lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)'
nproc
# Physical cores (unique core/socket pairs)
lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l
# CPU limit if running in a container ("max" means no limit)
cat /sys/fs/cgroup/cpu.max 2>/dev/null
free -g
```

From the output, set these two values for the rest of the steps:

```sh
export PHYSICAL=8    # physical cores
export LOGICAL=16    # logical CPUs (nproc)
export LABEL=my-machine-name
```

If the container CPU limit is lower than `LOGICAL`, note it; that limit is the
real ceiling.

## 3. Set up

```sh
# Rust (if not installed): https://rustup.rs
# uv (if not installed): https://docs.astral.sh/uv/
export UV_PROJECT_ENVIRONMENT=$HOME/uv-environments/forust
(cd py-forust && uv sync --group dev)
PY=$UV_PROJECT_ENVIRONMENT/bin/python

cargo build --release --example perf_phases

$PY scripts/make_perf_data.py --out /tmp/forust-perf/w200 --rows 250000 --eval-rows 62500 --cols 200
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w500 --rows 100000 --eval-rows 25000 --cols 500
```

Check the build reproduces the reference models (only meaningful if you also
copied `/tmp/forust-perf/golden` from the development VM; otherwise skip):

```sh
$PY scripts/perf_golden.py check
```

## 4. Tree-time sweep

Steady-state milliseconds per tree for each thread count, median of 3 runs.
The thread counts are half the physical cores, the physical cores, and all
logical CPUs. A serial run is included as a reference.

```sh
OUT=agent/feat/optimize/thread_results.jsonl
THREADS="$((PHYSICAL / 2)) $PHYSICAL $LOGICAL"

# Depth 5, 200 columns
$PY scripts/perf_sweep.py --data /tmp/forust-perf/w200 --rows 25000 100000 250000 \
  --threads $THREADS --iterations 40 --repeats 3 --label $LABEL --out $OUT

# Depth 8, 200 columns (many small nodes; most sensitive to thread count)
$PY scripts/perf_sweep.py --data /tmp/forust-perf/w200 --rows 25000 100000 \
  --threads $THREADS --iterations 20 --repeats 3 --label $LABEL --out $OUT -- --max-depth 8

# Depth 5, 500 columns
$PY scripts/perf_sweep.py --data /tmp/forust-perf/w500 --rows 25000 100000 \
  --threads $THREADS --iterations 30 --repeats 3 --label $LABEL --out $OUT
```

If `LOGICAL` equals `PHYSICAL` (no hyperthreading), also add `$((PHYSICAL * 2))`
to `THREADS` to confirm oversubscribing doesn't help.

## 5. End-to-end Python fit

This is what users see: a full `fit` from Python with a single evaluation set,
comparing the default with `num_threads` set to the physical and logical
counts.

```sh
$PY - <<'PY'
import os, sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
from make_perf_data import load
from forust import GradientBooster

physical, logical = int(os.environ["PHYSICAL"]), int(os.environ["LOGICAL"])
d = load(Path("/tmp/forust-perf/w200"))
for depth in (5, 8):
    for num_threads in (None, physical, logical):
        times = []
        for _ in range(2):
            model = GradientBooster(iterations=100, learning_rate=0.1, max_depth=depth,
                                    num_threads=num_threads)
            start = time.perf_counter()
            model.fit(d["X_train"], d["y_train"], evaluation_data=[(d["X_eval"], d["y_eval"])])
            times.append(time.perf_counter() - start)
        print(f"depth={depth} num_threads={num_threads!s:>4}: best of 2 = {min(times):6.1f} s", flush=True)
PY
```

`num_threads=None` and `num_threads=$LOGICAL` should take about the same time
(both use every logical CPU), which also confirms the setting is wired up.

## 6. What to send back

- The output of step 2.
- `agent/feat/optimize/thread_results.jsonl`, or the tables printed in step 4.
- The lines printed in step 5.

## How to read the results

- **Physical cores faster than logical by more than ~5%, consistently:**
  supports defaulting `num_threads` to the physical core count.
- **Within ~5% of each other:** keep the current default; document
  `num_threads` as an option.
- **Logical faster:** keep the current default.
- Depth 8 and 25k-row cases are the most sensitive; depth 5 at 250k rows the
  least. Repeat runs vary by about 5%, so treat smaller differences as noise.
