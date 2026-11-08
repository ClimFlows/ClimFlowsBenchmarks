#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=${JULIA:-julia}
backend=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
run_one() {
    local name=$1
    shift
    if [[ "$name" == shtns-jll ]]; then
        env -u SHTNS_PATH SHTNS_BINARY=jll "$julia_bin" --startup-file=no --project="$root/environments/$name" "$root/scripts/benchmark.jl" --backend=shtns "$@"
        return
    fi
    case "$name" in shtns|shtnskit|speedy) ;; *) echo "Unknown backend: $name" >&2; return 2;; esac
    [[ -f "$root/environments/$name/Manifest.toml" ]] || { echo "Run scripts/setup.sh first" >&2; return 2; }
    if [[ "$name" == shtns ]]; then
        : "${CUDA_PATH:?Set CUDA_PATH to the toolkit used to build SHTns}"
        export SHTNS_PATH=${SHTNS_PATH:-$root/build/shtns-c/libshtns_cuda.so}
        local cuda_libs="${CUDA_MATH_LIBS:+$CUDA_MATH_LIBS:}$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$CUDA_PATH/nvvm/lib64"
        LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$julia_bin" --startup-file=no --project="$root/environments/$name" "$root/scripts/benchmark.jl" --backend="$name" "$@"
    else
        "$julia_bin" --startup-file=no --project="$root/environments/$name" "$root/scripts/benchmark.jl" --backend="$name" "$@"
    fi
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
