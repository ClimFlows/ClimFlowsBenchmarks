#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root/sources"
sha256sum -c SHA256SUMS
mkdir -p "$root/vendor/legoesm-core"
tar -xzf legoesm-core.tar.gz -C "$root/vendor/legoesm-core"
test -f "$root/vendor/legoesm-core/legoesm/grids/gaussian.py"
