#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
uv_bin=${UV:-uv}
julia_bin=${JULIA:-julia}
"$uv_bin" sync --locked --project="$root/environments/torch"
JULIA="$julia_bin" "$uv_bin" run --locked --project="$root/environments/torch" \
    python "$root/scripts/benchmark_torch.py" "$@"
