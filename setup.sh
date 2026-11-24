#!/usr/bin/env bash
# Create a venv and install this repo. Works on a fresh GPU server.
#
#   bash setup.sh                 # CPU wheels (or whatever pip resolves)
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup.sh
#
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

if [ ! -d "$VENV" ]; then
  echo "==> creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip wheel

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  echo "==> installing torch/torchvision from $TORCH_INDEX_URL"
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

echo "==> installing the package (editable) + dev tools"
pip install -e '.[dev]'

python - <<'PY'
import torch, torchvision
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"cuda {torch.cuda.is_available()}")
PY

echo
echo "done. next:"
echo "  source $VENV/bin/activate"
echo "  pytest"
echo "  python scripts/smoke_test.py"
