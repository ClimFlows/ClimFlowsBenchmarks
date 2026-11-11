#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=${JULIA:-julia}
backend=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
run_one() {
    local name=$1
    shift
    if [[ "$name" == shtns-jll || "$name" == shtns-float32 ]]; then
        env -u SHTNS_PATH "$julia_bin" --startup-file=no --project="$root/environments/$name" "$root/scripts/benchmark.jl" --backend=shtns "$@"
        return
    fi
    case "$name" in shtnskit|speedy) ;; *) echo "Unknown backend: $name" >&2; return 2;; esac
    [[ -f "$root/environments/$name/Manifest.toml" ]] || { echo "Run scripts/setup.sh first" >&2; return 2; }
    "$julia_bin" --startup-file=no --project="$root/environments/$name" "$root/scripts/benchmark.jl" --backend="$name" "$@"
}
if [[ "$backend" == all ]]; then
    failed=0
    for name in shtns-jll shtnskit speedy; do
        run_one "$name" "$@" || failed=1
    done
    exit "$failed"
else
    run_one "$backend" "$@"
fi
