# CUDA Float32 validation, 10 November 2026

SHTns.jl local branch: `codex/cuda-float32`, commit
`d9981b8` (source snapshot and SHA256 in `sources/`).

Hardware: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU, compute
capability 12.0. Julia 1.11.9, CUDA runtime 12.9, SHTns_jll 3.7.0+0.

## Results

- Standalone prototype: all 10 benchmark cases pass (T21, T42, T63, T127,
  T255; Float32 and Float64; 64 tangential vector fields).
- Patched public SHTns.jl API: all 10 cases pass, 30 measured operations,
  60.000 GPU seconds and 449,018 executions.
- Every measured call allocates zero CPU and GPU bytes. CUDA graph outputs and
  post-measurement velocity reconstruction pass the existing checks.
- Maximum Float32 analytic relative L2 error: 1.4455886403723028e-4 (T127),
  below the unchanged 5e-4 tolerance. Float64 uses the unchanged 5e-10 tolerance.
- Maximum Float32 graph/direct relative L2 error: 5.50e-7, below 1e-5.
- CUDA package tests: 44 existing assertions + 936 Float32/Float64 assertions
  + 2 GPU-constructor assertions = 982 passed.
- The reporter validates all 30 retained historical JLL and in-place cases.
- All source-archive SHA256 checks pass.

## T255 tuning diagnostic

With C defaults, repeated direct Float32 vector analysis varies by about
2.6e-5 to 3.1e-5 relative L2, due to inter-block atomic summation. This exceeds
the benchmark's graph/direct check, independently of whether a graph is used.

The recorded T255 Float32 batch-64 run uses SHTns' existing tuning file:
`[4,4,2,6,4,12]`. A single 192-thread analysis block covers the northern
latitudes. Ten direct repeats and ten graph repeats were identical (zero
relative error). This profile is confined to the benchmark adapter and is
recorded in each affected result's notes. All data and transforms remain
Float32; the C library is the unmodified SHTns_jll binary.

An additional test used random coefficients covering the complete T255
spectrum for all 64 fields, with real m=0 and zero vector monopoles:

- Tangential synthesis vs CPU Float64: 7.1412e-7 and 6.6328e-7 relative L2.
- Spectral round trip: 1.1416e-6 and 1.1517e-6 relative L2.

## Reproduction

From `SphericalHarmonics`, with Julia 1.11.9 selected:

```sh
bash scripts/setup.sh shtns-float32
bash scripts/verify_shtns_float32.sh tests
bash scripts/verify_shtns_float32.sh prototype
bash scripts/verify_shtns_float32.sh diagnostic --full-spectrum
bash scripts/run.sh shtns-float32 --seconds-total=60 --out=results/cuda-float32
python3 scripts/report.py results/cuda-float32
```

The prototype uses the original pinned wrapper and direct Float32 C bindings.
The final run uses the archived package through its public API; no prototype
methods are loaded. The tuning profile is tested on the recorded GPU and
workload; other hardware may need separate tuning. No GitHub fork is required
for these local branches and archived sources.
