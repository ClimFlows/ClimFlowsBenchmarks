#!/usr/bin/env bash
# Fresh workspace and Julia depot; no fallback to a user's installed packages.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=${JULIA:-julia}
destination=${SHT_BENCH_REPRO_DIR:-$root/build/reproduction-jll-$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e "$destination" ]] || { echo "Destination must not exist: $destination" >&2; exit 2; }
mkdir -p "$destination"
destination=$(cd -- "$destination" && pwd)
work=$destination/workspace
mkdir -p "$work/sources" "$work/environments/shtns-jll" "$work/scripts" "$work/adapters"
cp "$root/scripts/"*.sh "$root/scripts/"*.jl "$root/scripts/"*.py "$work/scripts/"
cp "$root/adapters/"*.jl "$work/adapters/"
cp "$root/config.toml" "$work/"
cp "$root/sources/"*.tar.gz "$root/sources/SHA256SUMS" "$root/sources/revisions.toml" "$work/sources/"
cp "$root/environments/shtns-jll/"*.toml "$work/environments/shtns-jll/"
export JULIA_DEPOT_PATH=$destination/depot
export JULIA_LOAD_PATH='@:@stdlib'
export JULIA_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=2
export JULIA_PKG_SERVER=${SHT_BENCH_PKG_SERVER:-https://pkg.julialang.org}
unset SHTNS_PATH LD_LIBRARY_PATH LD_PRELOAD JULIA_PKG_OFFLINE
export JULIA=$julia_bin
"$julia_bin" --startup-file=no -e 'VERSION == v"1.11.9" || error("Reference Julia version: 1.11.9")'
printf '%s\n' "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH" "JULIA_LOAD_PATH=$JULIA_LOAD_PATH" "JULIA_PKG_SERVER=$JULIA_PKG_SERVER" > "$destination/environment.txt"
echo "Isolated reproduction directory: $destination"
# Preserve the fresh scripts, sources, manifests and logs even if setup fails.
bash "$work/scripts/setup.sh" shtns-jll > "$destination/setup.log" 2>&1
echo "Installation complete; starting GPU validation and measurements."
bash "$work/scripts/run.sh" shtns-jll --seconds-total=60 --out="$destination/results" "$@" > "$destination/run.log" 2>&1
python3 "$work/scripts/report.py" "$destination/results"
echo "Completed: $destination/results/summary.md"
