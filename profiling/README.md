# VRAM scaling — how far up the token axis each codebase gets

The top-level README claims that on one RTX Pro 6000, "OpenFold3 topped out at around
3500 residues while the AlphaFold 3 codebase reached ~5000". This directory is the
measurement behind that sentence: peak GPU memory against token count for the two
arms that differ only in the codebase, walked upwards until one of them stops.

Both sweeps have been run to completion, on one RTX Pro 6000 (96 GiB) per arm, 5
diffusion samples, 3 recycles, seed 42, templates off in both.

![Peak live GPU memory vs token count](results/vram_vs_tokens.png)

*Each curve ends at the largest pool that ran; the next 500-token step ran out of GPU
memory. Templates are off in both arms.*

| | largest pool that ran | peak live memory there | first failure |
|---|---|---|---|
| AlphaFold 3 code | **5500 tokens** | 79.0 GiB | GPU OOM at 6000 (reached 87.6 GiB) |
| OpenFold3 code | **3500 tokens** | 77.0 GiB | GPU OOM at 4000 (reached 92.5 GiB) |

The AF3 codebase runs the same weights on a pool **2000 tokens larger** on the same
card, and needs roughly half the memory at matched size (3500 tokens: 34.5 GiB against
77.0 GiB). Its ceiling here is the XLA arena, 0.95 x the card; OpenFold3 allocates on
demand and stops against the card itself.

Per-size numbers, including the nvidia-smi trace and the allocator's reserved
high-water, are in `results/af3_openbind_templateoff.tsv` and
`results/of3_openbind_templateoff.tsv` (copied out of
`<base>/profiling/<arm>/vram_sweep.tsv` by the plotting script). `--runtime` adds the model
time panel (`results/vram_and_runtime.png`), where the two engines cross over between
1000 and 1500 tokens — OpenFold3 is faster below that and slower above it, which is the
same crossover the Benchmark-IV runtimes show.

## What is measured

One arm per sweep, taken from the repository's own arm configs, so the sweep runs the
configuration the benchmark ran and not a variant of it:

| arm | config | codebase | weights |
|---|---|---|---|
| `af3_openbind_templateoff` | [`configs/af3_openbind_templateoff.json`](../configs/af3_openbind_templateoff.json) | AlphaFold 3 | OpenFold3 openbind |
| `of3_openbind_templateoff` | [`configs/of3_openbind_templateoff.json`](../configs/of3_openbind_templateoff.json) | OpenFold3 | OpenFold3 openbind |

Both templates-off, which is the pair the repository already treats as the like-for-like
code comparison: the OF3 converter carries no templates over, so the AF3 side has to
run without them too. Passing `configs/af3_openbind.json` instead sweeps the
templates-ON variant, and the difference between the two is the template memory cost.

## The inputs

`build_pool_inputs.py` builds one pool per token count: *n* copies of a single real
benchmark sequence, with that sequence's real MSAs and templates copied verbatim out of
`<base>/work/msa_store/`. Nothing is recomputed and nothing is synthetic — the monomer
is chosen by rule (shortest store sequence ≥ 150 aa whose length divides the 500-token
step), which on this dataset is a 250-aa chain with 1313 unpaired and 1274 paired
alignment rows, so every pool lands exactly on the 500-token grid.

The OF3 queries are then derived from those AF3 JSONs by the repository's own converter
([`scripts/09_build_of3_inputs.py`](../scripts/09_build_of3_inputs.py)), which is what
makes the two engines provably read the same alignments.

For protein-only inputs one standard residue is one token, so tokens == residues.

## Running it

```bash
export AF3OF3_BASE_DIR=/path/to/workdir/af3code_of3weights

# from a login node — one job per arm, each holding one GPU for the whole grid
bash profiling/run_sweep.sh --config configs/of3_openbind_templateoff.json --time 12:00:00
bash profiling/run_sweep.sh --config configs/af3_openbind_templateoff.json --time 12:00:00

# or inside an allocation that already has a GPU to itself
bash profiling/run_sweep.sh --config configs/of3_openbind_templateoff.json --local
```

The first invocation builds the pool inputs if they are missing. Add `--dry-run` to see
the `sbatch` line without submitting, `--max-tokens N` to stop the grid early, and
`--end`/`--step` to change the grid that gets built.

Each job appends one row per size to `<base>/profiling/<arm>/vram_sweep.tsv` as it
finishes, and stops at the first size that fails. A re-submission skips sizes already
in that table, so a job that runs out of wall clock is resumed by submitting it again.

Then plot (matplotlib comes from the pixi environment):

```bash
pixi run python profiling/plot_vram.py --out profiling/results/vram_vs_tokens.png
pixi run python profiling/plot_vram.py --runtime --out profiling/results/vram_and_runtime.png
```

`plot_vram.py` picks up every `<arm>/vram_sweep.tsv` under `$AF3OF3_BASE_DIR/profiling`
(override with `--sweep-dir`, restrict and order with `--arms`), prints the per-size
table it plotted, and takes `--metric peak_vram_gib` to plot the nvidia-smi trace
instead of the default live-buffer high-water.

## Two measurement traps

Both were hit on the first attempt, and both are why the numbers are collected the way
they are now.

**1. Turning off JAX preallocation moves the boundary.** `nvidia-smi` cannot see inside
a preallocated arena, so the obvious way to get a memory curve out of AlphaFold 3 is to
set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and let the device reading follow demand. That
is wrong twice over: it caps the arena at JAX's default `0.75` fraction, and it leaves
BFC growing in separate regions, so a single large block cannot be placed in any of
them. Measured on the 4000-token pool:

| AF3 allocator env | 4000 tokens |
|---|---|
| `preallocate=false` (default 0.75 fraction) | OOM — could not allocate one 41.62 GiB block, device at 50.6 GiB |
| `preallocate=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | OOM — same 41.62 GiB block |
| `preallocate=true`, `XLA_CLIENT_MEM_FRACTION=0.95` | ran, 476 s |

So the sweep uses the third row, which is what the openbind runs use
(`pooled_ppi_comparison/scripts/2026aug_af3_openbind_weights`). `MEM_FRACTION` changes
the fraction; `XLA_PYTHON_CLIENT_MEM_FRACTION` must be left alone, because jaxlib
refuses to start when both fraction variables are set.

**2. Under preallocation, `nvidia-smi` reports the arena, not the demand** — 93 GiB flat
at every size. So the curve comes from the frameworks' own counters instead
(`peak_mem.py`, which wraps the engine's entry point):

| column | what it is |
|---|---|
| `peak_alloc_mib` | live-buffer high-water: JAX `peak_bytes_in_use`, PyTorch `max_memory_allocated`. **The plotted curve** — the same quantity for both engines, independent of allocator policy |
| `peak_reserved_mib` | what the allocator held from the driver (AF3: the arena) |
| `peak_vram_mib` | `nvidia-smi` whole-device high-water, polled once a second |

The counters are read in a `finally` block, not an `atexit` hook: `atexit` runs
last-registered-first, so a hook registered before the engine imports its framework
runs *after* that framework has torn its backend down and reads zeroes. It did, first
time round.

`peak_vram_mib` is whole-device, so it also counts any other tenant of the same GPU —
the sweep is only meaningful with the card to itself. `gpu_total_mib` is in every row so
the ceiling is on the record.

`status` is not taken from the exit code alone: OpenFold3 exits 0 having failed every
query — it reports `OOM for query_id(s) ...` in its log and `Failed Queries: 1` in its
summary — and both engines print an out-of-memory before dying, so the log is read too.
`gpu_oom` is the boundary the sweep is looking for; `cpu_oom` (exit 137, the host OOM
killer) is not a GPU limit and is reported separately.

## Other things that change the answer

* **`TF_FORCE_UNIFIED_MEMORY` is deliberately left unset.** AlphaFold 3 recommends it
  above ~5120 tokens; it works by spilling to host RAM, which would turn the GPU
  boundary into a much slower run that never OOMs.
* **AlphaFold 3 pads token counts to buckets** (512, 1024, … 5120 by default), so its
  curve is a staircase and a 500-token pool is really a 512-token forward pass.
  OpenFold3 does not bucket.
* **The pools are homomeric.** A pool of one repeated sequence is not a biological
  complex and its predicted structure is meaningless — nothing here is scored. It is a
  token-count dial with a realistic feature payload behind it. Both engines expand the
  same entity *n* times, and the pool carries no cross-species pairing signal that a
  real heterocomplex would.
* **`num_diffusion_samples` is 5, from the arm configs.** Diffusion samples are batched,
  so peak memory depends on it; a sweep at 1 sample would reach further.

## Files

| file | what it does |
|---|---|
| `build_pool_inputs.py` | one AF3 input JSON per token count, from the MSA store |
| `run_sweep.sh` | read an arm config, build inputs if needed, submit or run the sweep |
| `sweep.sbatch` | hold one GPU, walk the grid upwards, stop at the first failure |
| `run_with_trace.py` | poll `nvidia-smi` around one prediction, classify the outcome |
| `peak_mem.py` | wrap the engine's entry point, record its own peak-memory counters |
| `plot_vram.py` | peak VRAM (and optionally model time) vs tokens, per arm |
