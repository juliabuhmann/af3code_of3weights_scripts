#!/usr/bin/env bash
# Build the environment the `af3kit_*` arms run in: Anthropic's `af3_jax` optimization kit on the
# AlphaFold 3 JAX codebase, WITHOUT a container and WITHOUT compiling the C++ extension.
#
#   AF3OF3_KIT_ROOT=/path/to/anthropic_kits bash profiling/setup_af3kit_env.sh
#
# WHY NOT THE KIT'S OWN ROUTES. Route A/B need Docker to build the image (none on Euler; no
# prebuilt image published as of 2026-09-21); route C compiles the alphafold3 wheel from source
# (CMake, abseil, libcifpp, dssp -- a toolchain the compute nodes lack). But the kit pins
# sokrypton/alphafold3 v3.1.4 (bc32b22f), and a COMPILED alphafold3_open 3.1.4 wheel of exactly
# that commit sits in the host's uv cache (cpp.cpython-312 extension included). The stack the fork
# pins (jax 0.10.2, dm-haiku 0.0.16, tokamax 0.0.12) is what the kit pins too. So the venv is:
# the pinned runtime from PyPI + the cached compiled package + upstream's build_data step + the
# kit's two packages on PYTHONPATH (its `-m af3_jax_opt` route activates from the source trees;
# no install). When a container image exists, prefer it: the arm's launcher block then names the
# .sif and this file is unnecessary.
#
# Lays out, under $AF3OF3_KIT_ROOT:
#   uplifting-biomolecular-modeling/   the kit release, git clone at KIT_COMMIT (nothing vendored)
#   venv/af3-jax                       the venv; interpreter under venv/uv-python (never $HOME)
#   af3_repo/                          WRITABLE copy of the fork's run_alphafold.py + converter:
#                                      a kit mode copies run_alphafold_fast.py beside them
#   weights/af3_jax/p2/                the preview-2 weights, fetched + converted + sha256-checked
#                                      by the kit's own installer
#
# Three things the kit's Dockerfile does that a bare venv must repeat, each hit on first try:
#   1. LIBCIFPP_DATA_DIR: the wheel's CCD (share/libcifpp/components.cif) is not on libcifpp's
#      compiled-in search path here; the arm exports it (configs/af3kit_*.json launcher.env).
#   2. build_data: upstream's post-install step that writes the CCD pickles into the package.
#   3. torch (CPU): the kit's converter reads the PyTorch checkpoint; the kit pins torch 2.7.1+cpu.
# The openbind fork's four Python-only changes are overlaid too (they are inert for p2 weights;
# they gate on of3_openbind), so ONE venv serves both weight variants.
set -euo pipefail
ROOT="${AF3OF3_KIT_ROOT:?set AF3OF3_KIT_ROOT to the directory the environment goes under}"
KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e     # "Initial public release", 2026-09-17
FORK_REPO=https://github.com/juliabuhmann/alphafold3
FORK_BRANCH=add_openbind_porter                            # v3.1.4 + the openbind porter (1fff5c4)
CACHED_WHEEL_DIR="${AF3OF3_AF3_CACHED:-/cluster/project/beltrao/jbuhmann/pixi/cache/uv-cache/archive-v0/JvQUuu8wwM4Q5c8d}"   # alphafold3_open 3.1.4, compiled
KIT="$ROOT/uplifting-biomolecular-modeling/af3_jax"
VENV="$ROOT/venv/af3-jax"
export UV_PYTHON_INSTALL_DIR="$ROOT/venv/uv-python" UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uv" PIP_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/pip"

echo "=== [1/6] kit release at $KIT_COMMIT"
if [ ! -d "$ROOT/uplifting-biomolecular-modeling/.git" ]; then
    git clone -q "$KIT_REPO" "$ROOT/uplifting-biomolecular-modeling"; fi
git -C "$ROOT/uplifting-biomolecular-modeling" checkout -q "$KIT_COMMIT"
[ -f "$ROOT/uplifting-biomolecular-modeling/af3_jax/stock/PINS.json" ] || { echo "ERROR: not the kit tree" >&2; exit 1; }
# the openbind layout support (profiling/patches/af3_jax_openbind.diff); applied once, idempotent
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/af3_jax_openbind.diff"
if ! git -C "$ROOT/uplifting-biomolecular-modeling" apply --check --reverse "$PATCH" 2>/dev/null; then
    git -C "$ROOT/uplifting-biomolecular-modeling" apply "$PATCH" && echo "  applied af3_jax_openbind.diff"
else echo "  af3_jax_openbind.diff already applied"; fi

echo "=== [2/6] venv (python 3.12 = the cached wheel's ABI) + the pinned runtime"
[ -x "$VENV/bin/python" ] || uv venv --seed --managed-python --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q "jax[cuda12]==0.10.2" "dm-haiku==0.0.16" "tokamax==0.0.12" "rdkit==2025.9.4" \
    "absl-py>=2.3.1" "etils[epath]" numpy tqdm zstandard
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"      # deviation 3

echo "=== [3/6] the compiled alphafold3 3.1.4 package from the uv cache"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
[ -d "$CACHED_WHEEL_DIR/alphafold3" ] || { echo "ERROR: no cached alphafold3_open 3.1.4 at $CACHED_WHEEL_DIR (AF3OF3_AF3_CACHED)" >&2; exit 1; }
[ -d "$SP/alphafold3" ] || { cp -r "$CACHED_WHEEL_DIR/alphafold3" "$CACHED_WHEEL_DIR"/alphafold3_open-3.1.4.dist-info "$SP/"
    for d in share etc lib include var; do [ -d "$CACHED_WHEEL_DIR/$d" ] && cp -rn "$CACHED_WHEEL_DIR/$d" "$VENV/" 2>/dev/null || true; done; }
export LIBCIFPP_DATA_DIR="$VENV/share/libcifpp"                                                   # deviation 1
python -c "import alphafold3, alphafold3.cpp"

echo "=== [4/6] the fork's openbind porter (4 Python files) + a writable repo copy"
FORK="$ROOT/alphafold3-fork"
[ -d "$FORK/.git" ] || { git clone -q --filter=blob:none https://github.com/sokrypton/alphafold3 "$FORK"; git -C "$FORK" remote add jb "$FORK_REPO"; }
git -C "$FORK" fetch -q jb
for f in model/model_config.py model/network/modules.py model/network/diffusion_transformer.py model/of3_weight_converter.py; do
    git -C "$FORK" show "jb/$FORK_BRANCH:src/alphafold3/$f" > "$SP/alphafold3/$f"; done
mkdir -p "$ROOT/af3_repo"
git -C "$FORK" show "jb/$FORK_BRANCH:run_alphafold.py" > "$ROOT/af3_repo/run_alphafold.py"
git -C "$FORK" show "jb/$FORK_BRANCH:convert_of3_weights.py" > "$ROOT/af3_repo/convert_of3_weights.py"
[ -f "$SP/alphafold3/constants/converters/ccd.pickle" ] || python -c "from alphafold3.build_data import build_data; build_data()"   # deviation 2

echo "=== [5/6] the kit's weights: fetch, convert, digest-check (the kit's own installer)"
export PYTHONPATH="$KIT/opt:$KIT/../common/opt_core" MODEL_OPT="$KIT" AF3_JAX_PY="$VENV/bin/python" AF3_JAX_REPO="$ROOT/af3_repo" PYTHONHASHSEED=0
mkdir -p "$ROOT/weights/af3_jax"
( cd "$KIT" && bash run.sh install --config a100 --weights "$ROOT/weights/af3_jax" ) | grep -E "CONVERT|WEIGHTS|NOT MET" || true

echo "=== [6/6] verify"
python - <<'PY'
import inspect
from alphafold3.model.network import modules, diffusion_transformer
assert "of3_openbind" in inspect.getsource(modules.GridSelfAttention)
assert "of3_openbind" in inspect.getsource(diffusion_transformer.Transformer)
import af3_jax_opt, opt_core; print("af3_jax_opt", af3_jax_opt.__version__, "opt_core", opt_core.__version__)
print("openbind porter live; kit importable")
PY
cat <<EOT

paths for configs/af3kit_*.json launcher block:
  kit_dir     : $KIT
  kit_env     : $VENV
  kit_repo    : $ROOT/af3_repo
  model_dir   : $ROOT/weights/af3_jax/p2
  env         : LIBCIFPP_DATA_DIR=$VENV/share/libcifpp
EOT
