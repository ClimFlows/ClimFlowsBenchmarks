#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=${JULIA:-julia}
cd "$root/sources"
sha256sum -c SHA256SUMS
mkdir -p "$root/vendor"
for name in shtns-julia shtns-julia-float32 shtnskit speedy; do
    if [[ ! -d "$root/vendor/$name" ]]; then
        tar -xzf "$name.tar.gz" -C "$root/vendor"
    fi
done
export JULIA_PKG_PRECOMPILE_AUTO=0
for backend in "$@"; do
    case "$backend" in shtns-jll|shtns-float32|shtnskit|speedy) ;; *) echo "Unknown backend: $backend" >&2; exit 2;; esac
done
backends=("$@")
if [[ ${#backends[@]} == 0 ]]; then backends=(shtns-jll shtnskit speedy); fi
for backend in "${backends[@]}"; do
    cd "$root/environments/$backend"
    "$julia_bin" --startup-file=no --project=. "$root/scripts/instantiate.jl"
done
