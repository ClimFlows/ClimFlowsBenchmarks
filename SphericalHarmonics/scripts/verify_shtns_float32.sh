#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export JULIA=${JULIA:-${HOME}/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/tmp/climflows-sht-bench-depot:${HOME}/.julia}
stage=${1:-prototype}
case "$stage" in
prototype)
    export SHTNS_BENCH_ADAPTER="$root/adapters/shtns_float32_prototype.jl"
    bash "$root/scripts/run.sh" shtns-jll --seconds-total=60 --out="$root/results/cuda-float32-prototype"
    ;;
package)
    unset SHTNS_BENCH_ADAPTER
    env -u SHTNS_PATH SHTNS_BINARY=jll "$JULIA" --startup-file=no --project="$root/environments/shtns-float32" "$root/scripts/benchmark.jl" --backend=shtns --seconds-total=60 --out="$root/results/cuda-float32"
    ;;
diagnostic)
    env -u SHTNS_PATH SHTNS_BINARY=jll "$JULIA" --startup-file=no --project="$root/environments/shtns-jll" "$root/scripts/diagnose_shtns_float32.jl" "${@:2}"
    ;;
tests)
    env -u SHTNS_PATH "$JULIA" --startup-file=no --project="$root/environments/shtns-float32" "$root/vendor/shtns-julia-float32/test/ext/cudaext.jl"
    ;;
*) echo "Expected prototype, package, diagnostic or tests" >&2; exit 2 ;;
esac
