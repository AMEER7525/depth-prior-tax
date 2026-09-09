#!/usr/bin/env bash
# Set up the GPU host (4080 box or Colab). Verifies CUDA before installing
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
echo "== warming gsplat JIT (first import compiles CUDA kernels, can take minutes) =="
python -c "
import torch, gsplat
print('gsplat', gsplat.__version__)
from gsplat import rasterization
print('rasterization imported OK')
"

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
Split across hosts with --shard 0/2 (4080) and --shard 1/2 (Colab).
EOF
