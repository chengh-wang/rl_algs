#!/usr/bin/env bash
set -euo pipefail

MBDAC_ROOT=$(cd "$(dirname "$0")/.." && pwd)
AMP_SOURCE=${1:-"$MBDAC_ROOT/external/AMP_mjlab"}
AMP_COMMIT=6c7a2947fccc973e4af8e6d90e550400f1b6fcfc
if [[ ! -d "$AMP_SOURCE/.git" ]]; then
  mkdir -p "$(dirname "$AMP_SOURCE")"
  git clone https://github.com/ccrpRepo/AMP_mjlab.git "$AMP_SOURCE"
  git -C "$AMP_SOURCE" checkout "$AMP_COMMIT"
fi
test "$(git -C "$AMP_SOURCE" rev-parse HEAD)" = "$AMP_COMMIT"
git -C "$AMP_SOURCE" diff --quiet
git -C "$AMP_SOURCE" diff --cached --quiet

if [[ ! -f "$MBDAC_ROOT/.venv-amp/pyvenv.cfg" ]]; then
  uv venv --python 3.11 "$MBDAC_ROOT/.venv-amp"
fi
AMP_PYTHON="$MBDAC_ROOT/.venv-amp/bin/python"
uv pip install --python "$AMP_PYTHON" -r "$MBDAC_ROOT/requirements-mjlab.lock.txt"
# The upstream benchmark explicitly uses its RSL fork. Isolate it from the core env.
uv pip install --python "$AMP_PYTHON" --no-deps -e "$AMP_SOURCE" -e "$AMP_SOURCE/rsl_rl" -e "$MBDAC_ROOT"
"$AMP_PYTHON" - "$AMP_SOURCE" <<'PY'
from pathlib import Path
import shutil
import sys
import mjlab
source=Path(sys.argv[1])/'mjlab_patch/mjlab/managers/observation_manager.py'
target=Path(mjlab.__file__).parent/'managers/observation_manager.py'
backup=target.with_suffix('.py.upstream-original')
if not backup.exists(): shutil.copyfile(target,backup)
shutil.copyfile(source,target)
print('Applied the upstream observation patch only to the isolated benchmark environment.')
PY
