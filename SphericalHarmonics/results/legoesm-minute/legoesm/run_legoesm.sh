#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
uv_bin=${UV:-uv}
bash "$root/scripts/setup_legoesm.sh"
"$uv_bin" sync --locked --project="$root/environments/legoesm"
PYTHONPATH="$root/vendor/legoesm-core${PYTHONPATH:+:$PYTHONPATH}" \
    JAX_ENABLE_X64=1 \
    "$uv_bin" run --locked \
    --project="$root/environments/legoesm" python "$root/scripts/benchmark_legoesm.py" "$@"
