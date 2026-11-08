# ClimFlowsBenchmarks

Numerical-kernel benchmarks for [ClimFlows](https://github.com/ClimFlows).
Each benchmark lives in its own directory and includes its configuration,
dependency manifests, validation procedure and reference results.

| Benchmark | Scope | Documentation |
|---|---|---|
| SphericalHarmonic | CUDA scalar/vector spherical harmonic transforms; batch 64; Float32/Float64; T21–T255 | [Protocol and reproduction](SphericalHarmonic/README.md) |

## Spherical harmonic transforms

The current comparison covers SHTns.jl, SHTnsKit.jl and SpeedyTransforms.jl.
The SHTns comparison includes the precompiled SHTns_jll binary with CUDA 12.9
and a locally built SHTns library with CUDA 13.1.

- [Precompiled SHTns results](SphericalHarmonic/results/jll-cuda12.9/summary.md)
- [Initial three-library comparison](SphericalHarmonic/results/inplace-minute/summary.md)
- [Fresh-depot reproduction status](SphericalHarmonic/results/jll-cuda12.9/reproduction/STATUS.md)

The GPU measurements passed numerical validation. Installation from a fresh
Julia depot has been attempted but is **not yet verified end to end**, because
CUDA artifact downloads encountered decompression errors. These are separate
claims; a recorded timing does not imply a successful clean installation.

From `SphericalHarmonic/`, with Julia 1.11.9 and a compatible NVIDIA driver:

```sh
export JULIA=/path/to/julia-1.11.9/bin/julia
bash scripts/reproduce_jll.sh
```

This creates a new workspace and a new Julia depot, installs the pinned
dependencies using Julia's package manager, validates the transforms, and
measures approximately 60 seconds of GPU execution. Initial installation
requires network access and several GB of disk space.

## Provenance

Upstream source snapshots and their original license files are included in
`SphericalHarmonic/sources/`; commits and checksums are recorded alongside them.
Generated caches and native build products are excluded from version control.
Reference-result files retain the original paths and execution metadata for
traceability; these paths are not used by the reproduction scripts.
