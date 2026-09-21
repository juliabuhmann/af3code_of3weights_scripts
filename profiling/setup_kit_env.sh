#!/usr/bin/env bash
# Build the environment the `of3kit_*` arms run in: Anthropic's `openfold3_ob0` optimization kit
# (OpenFold3 0.5.0 code + the OpenBind-0 checkpoint) in a Python venv, following the kit's own
# route C (its README 'Setup' and STOCK.md 'Stack'), with the deviations this host forces.
#
#   AF3OF3_KIT_ROOT=/path/to/anthropic_kits bash profiling/setup_kit_env.sh
#
# Lays out, under $AF3OF3_KIT_ROOT:
#   uplifting-biomolecular-modeling/   the kit release, git clone at KIT_COMMIT (nothing is vendored here)
#   venv/of3-ob0                       the venv; interpreter under venv/uv-python (never $HOME)
#   conda_envs/cuda-nvcc               an nvcc, see deviation 2
#   conda_envs/x11 + lib-x11/          X11 client libs, see deviation 3
#   weights/of3-ob-2025-06-30-174k.pt  the checkpoint, fetched and sha256-checked by the kit
# and the configs/of3kit_*.json `launcher` block names exactly these paths.
#
# Deviations from the kit's route C, each forced by the host (Euler: no root, no CUDA toolkit):
#   1. DeepSpeed is installed WITHOUT DS_BUILD_EVOFORMER_ATTN=1. That flag compiles the DS4Sci
#      evoformer-attention op ahead of time and needs nvcc. The kit's stock configuration and
#      every mode keep that op OFF (its STOCK.md 'Stack'), so no path used here loads it.
#   2. `import deepspeed` still probes $CUDA_HOME/bin/nvcc for a version string, so a conda env
#      holding only cuda-nvcc at the stack's CUDA version supplies the binary. Nothing is compiled.
#   3. RDKit links libXrender/libXext/libSM; the kit lists them as a route-C host prerequisite.
#      They come from conda, and only the needed .so files are symlinked into lib-x11/ so the
#      conda env's own libstdc++ cannot shadow the venv's.
#
# Everything is pinned: the kit commit, the kit's requirements.lock, its wheel, its checkpoint digest.
# Rerunnable; each step is skipped when its product is present. Built on Euler, 2026-09-18.
set -euo pipefail
ROOT="${AF3OF3_KIT_ROOT:?set AF3OF3_KIT_ROOT to the directory the environment goes under}"
KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e     # "Initial public release of the model-optimization kits", 2026-09-17
CUDA_VER=12.8                                              # = the kit stack's torch build (torch 2.10.0+cu128)
KIT="$ROOT/uplifting-biomolecular-modeling/openfold3_ob0"
VENV="$ROOT/venv/of3-ob0"
export UV_PYTHON_INSTALL_DIR="$ROOT/venv/uv-python"
export UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uv" PIP_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/pip"
mkdir -p "$ROOT"

echo "=== [1/7] kit release at $KIT_COMMIT"
if [ ! -d "$ROOT/uplifting-biomolecular-modeling/.git" ]; then
    git clone -q "$KIT_REPO" "$ROOT/uplifting-biomolecular-modeling"
fi
git -C "$ROOT/uplifting-biomolecular-modeling" checkout -q "$KIT_COMMIT"

echo "=== [2/7] nvcc $CUDA_VER (deviation 2)"
[ -x "$ROOT/conda_envs/cuda-nvcc/bin/nvcc" ] || mamba create -y -q -p "$ROOT/conda_envs/cuda-nvcc" -c nvidia "cuda-nvcc=$CUDA_VER"
export CUDA_HOME="$ROOT/conda_envs/cuda-nvcc"

echo "=== [3/7] X11 client libraries (deviation 3)"
[ -d "$ROOT/conda_envs/x11" ] || mamba create -y -q -p "$ROOT/conda_envs/x11" -c conda-forge xorg-libxrender xorg-libxext xorg-libsm xorg-libx11 xorg-libice
mkdir -p "$ROOT/lib-x11"
for so in libXrender.so.1 libXext.so.6 libSM.so.6 libICE.so.6 libX11.so.6 libxcb.so.1 libXau.so.6 libXdmcp.so.6; do
    [ -e "$ROOT/conda_envs/x11/lib/$so" ] && ln -sf "$ROOT/conda_envs/x11/lib/$so" "$ROOT/lib-x11/$so"
done
export LD_LIBRARY_PATH="$ROOT/lib-x11${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== [4/7] venv from the kit's requirements.lock (all but openfold3 and deepspeed)"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --seed --managed-python --python 3.11 "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    grep -v -E '^(#|openfold3==|deepspeed==)' "$KIT/environment/requirements.lock" > "$ROOT/stack.txt"
    python -m pip install -q --no-deps -r "$ROOT/stack.txt"
    (cd "$KIT" && python -I stock/install_upstream.py --wheel-only)      # the pinned openfold3 0.5.0 wheel, sha256-checked
    python -m pip install -q --no-deps deepspeed==0.19.2                  # deviation 1
    python -m pip check
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "=== [5/7] kit packages, pin check, CCD, checkpoint (the kit's own installer)"
mkdir -p "$ROOT/weights"
(cd "$KIT" && bash run.sh install --weights "$ROOT/weights")

echo "=== [6/7] verify"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import openfold3, deepspeed, rdkit.Chem; print('openfold3 + deepspeed + rdkit import OK')"
export OPENFOLD3_OB0_CKPT="$ROOT/weights/of3-ob-2025-06-30-174k.pt"
(cd "$KIT" && bash run.sh check --config h100 --mode exact >/dev/null && bash run.sh check --config h100 --mode fast >/dev/null && echo "kit check: exact, fast dry-run OK")

echo "=== [7/7] paths for configs/of3kit_*.json"
cat <<PATHS
  "kit_dir":         "$KIT",
  "kit_commit":      "$KIT_COMMIT",
  "kit_env":         "$VENV",
  "cuda_home":       "$ROOT/conda_envs/cuda-nvcc",
  "ld_library_path": "$ROOT/lib-x11",
  "ckpt":            "$OPENFOLD3_OB0_CKPT"
PATHS
