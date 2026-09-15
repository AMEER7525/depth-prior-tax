#!/usr/bin/env bash
# Set up the GPU host (Colab). Verifies CUDA before installing
# gsplat, because a CUDA-less failure from pip is deeply unhelpful.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== environment =="
python -c "
import sys, torch
print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__)
print('cuda    ', torch.version.cuda, '| available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu     ', p.name, f'{p.total_memory/1e9:.1f} GB, sm_{p.major}{p.minor}')
else:
    raise SystemExit('ERROR: no CUDA GPU. gsplat cannot be built or run here.')
"

echo
echo "== installing =="
pip install -r requirements-gpu.txt
# gsplat compiles kernels on first import; do it now so a 200-run sweep does not
# stall on it (and so a build failure surfaces during setup, not at 2am).
echo
# gsplat compiles its CUDA kernels on the first RENDER, not on import, so a
# bare import proves nothing and leaves the compile (and its spinner, which
# prints hundreds of lines in a notebook) to the first training run. Render
# one Gaussian here instead, with the build log sent to a file. The result is
# cached under $TORCH_EXTENSIONS_DIR (on Drive, per GPU type), so this takes
# minutes once per GPU type and seconds afterwards.
echo "== gsplat CUDA kernels (first time on a GPU type: ~5 min; cached after) =="
LOG="${TMPDIR:-/tmp}/gsplat_build.log"
if python - >"$LOG" 2>&1 <<'PY'
import torch, gsplat
from gsplat import rasterization
dev = "cuda"
viewmat = torch.eye(4, device=dev)[None]
viewmat[0, 2, 3] = 3.0
K = torch.tensor([[[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]]], device=dev)
_, alpha, _ = rasterization(
    torch.zeros(1, 3, device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev),
    torch.full((1, 3), 0.1, device=dev), torch.ones(1, device=dev),
    torch.ones(1, 3, device=dev), viewmat, K, 64, 64)
torch.cuda.synchronize()
print(f"gsplat {gsplat.__version__}: CUDA kernels OK (test render alpha max {alpha.max().item():.2f})")
PY
then
  tail -n 1 "$LOG"
else
  echo "gsplat CUDA build/render FAILED -- last lines of $LOG:"
  tail -n 25 "$LOG"
  exit 1
fi

echo
echo "== data root =="
: "${DATA_ROOT:=data}"
echo "DATA_ROOT=$DATA_ROOT"
for d in "$DATA_ROOT/nerf_synthetic" "$DATA_ROOT/dtu"; do
  [ -d "$d" ] && echo "  found  $d" || echo "  MISSING $d"
done

echo
echo "== tests =="
python -m pytest tests/ -q

cat <<'EOF'

Next:
  export DATA_ROOT=/path/to/data RUNS_ROOT=/path/to/runs
  python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage1 --resume
Split a stage across sessions with --shard 0/2 and --shard 1/2.
EOF
