#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=${JULIA:-julia}
: "${CUDA_PATH:?Set CUDA_PATH to a CUDA toolkit supporting your GPU (Blackwell: CUDA 12.8 or newer)}"
export PATH="$CUDA_PATH/bin:$PATH"
command -v nvcc >/dev/null
[[ -d "$root/vendor/shtns-c" ]] || { echo "Run scripts/setup.sh shtns first" >&2; exit 2; }
fftw_root=$("$julia_bin" --startup-file=no --project="$root/environments/shtns" -e 'using FFTW_jll; print(FFTW_jll.artifact_dir)')
mkdir -p "$root/build/shtns-c"
cp -a "$root/vendor/shtns-c/." "$root/build/shtns-c/"
cd "$root/build/shtns-c"
export CPPFLAGS="-I$fftw_root/include ${CPPFLAGS:-}"
# Upstream puts -l flags before object files: retain them with GNU ld --as-needed defaults.
export LDFLAGS="-Wl,--no-as-needed -L$fftw_root/lib -Wl,-rpath,$fftw_root/lib -Wl,-rpath,$CUDA_PATH/lib64 ${LDFLAGS:-}"
# Explicit compute capability avoids old architecture probes, removed in recent toolkits.
./configure --enable-cuda="${CUDA_ARCH:-120}" --enable-shared --disable-openmp 2>&1 | tee configure.log
make -j"${BUILD_JOBS:-2}" 2>&1 | tee make.log
[[ -f libshtns_cuda.so ]]
sha256sum libshtns_cuda.so > SHA256SUMS
nvcc --version > nvcc-version.txt
"${CC:-cc}" --version > cc-version.txt
echo "Built $root/build/shtns-c/libshtns_cuda.so"
