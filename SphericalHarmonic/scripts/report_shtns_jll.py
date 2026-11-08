"""Summarize and verify a SHTns_jll run; optional local-build comparison."""
import pathlib
import sys
import tomllib
import math

root = pathlib.Path(sys.argv[1])
data = root / "shtns"
meta = tomllib.loads((data / "metadata.toml").read_text())
assert meta["shtns_binary_origin"] == "jll"
assert meta["shtns_library_sha256"] == "11097a94101be9941966ac91e1cd2193b78799b23bcfd71f5445d51e203349ca"
lines = ["# SHTns_jll avec CUDA " + meta["cuda_runtime_version"], "",
         "Binaire SHTns_jll 3.7.0+0 inchangé (variante CUDA 12.6), runtime CUDA "
         + meta["cuda_runtime_version"] + ", batch 64.", "",
         "| Troncature | Statut Float64 | Aller-retour GPU (ms) | Compilation locale CUDA 13.1 (ms) |",
         "|---|---|---:|---:|"]
gpu_seconds = executions = samples = 0
failures = []
for degree in (21, 42, 63, 127, 255):
    r = tomllib.loads((data / f"T{degree}-Float64.toml").read_text())
    assert r["lmax"] == degree and r["batch"] == 64
    unsupported = tomllib.loads((data / f"T{degree}-Float32.toml").read_text())
    assert unsupported["status"] == "unsupported"
    timing = "—"
    if r["status"] == "ok":
        assert max(r["errors"].values()) <= r["tolerance"]
        assert r["post_measurement_velocity_error"] <= r["tolerance"]
        for op in ("analysis", "synthesis", "roundtrip"):
            s = r[op]
            assert s["host_allocated_bytes_per_call"] == s["gpu_allocated_bytes_per_call"] == 0
            assert s["direct_gpu_allocated_bytes_per_call"] == 0
            assert s["samples"] >= 10
            assert all(math.isfinite(t) and t > 0 for t in s["seconds"])
            gpu_seconds += s["gpu_measured_seconds"]
            executions += s["executions"]
            samples += s["samples"]
        timing = f'{1000*r["roundtrip"]["median_seconds"]:.4f}'
    else:
        failures.append(degree)
    reference = root.parent / "inplace-minute" / "shtns" / f"T{degree}-Float64.toml"
    local = tomllib.loads(reference.read_text()) if reference.exists() else None
    baseline = f'{1000*local["roundtrip"]["median_seconds"]:.4f}' if local else "—"
    lines.append(f'| T{degree} | {r["status"]} | {timing} | {baseline} |')
lines += ["", f"Calcul GPU mesuré : {gpu_seconds:.3f} s ; {executions} exécutions ; {samples} échantillons.",
          "", "Temps médians par batch de 64 champs vectoriels ; buffers préalloués et relecture de graphes CUDA.",
          "Les références locales proviennent de la campagne antérieure, sans exécution simultanée.",
          "Float32 non pris en charge par l'API CUDA du wrapper Julia retenu.", "",
          "## Bibliothèques chargées", ""]
lines += [f"- `{p}`" for p in meta["loaded_cuda_libraries"]]
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:15]))
if failures:
    sys.exit(f"Cases failed: {failures}; timings are incomplete")
