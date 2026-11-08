#!/usr/bin/env python3
"""Validate and summarize the complete allocation-free, one-minute campaign."""
import collections
import math
import pathlib
import sys
import tomllib

root = pathlib.Path(__file__).resolve().parents[1]
folder = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/inplace-minute"
config = tomllib.loads((folder / "shtns/config.toml").read_text())
data = {}
for backend in ("shtns", "shtnskit", "speedy"):
    for case in config["cases"]:
        degree = case["lmax"]
        for precision in ("Float32", "Float64"):
            path = folder / backend / f"T{degree}-{precision}.toml"
            result = tomllib.loads(path.read_text())
            if backend == "shtns" and precision == "Float32":
                assert result["status"] == "unsupported", path
                continue
            assert result["status"] == "ok", (path, result.get("error", result["status"]))
            assert result["mode"] == "inplace" and result["batch"] == 64, path
            assert max(result["errors"].values()) <= result["tolerance"], path
            assert result["post_measurement_velocity_error"] <= result["tolerance"], path
            for operation in ("analysis", "synthesis", "roundtrip"):
                stats = result[operation]
                assert stats["gpu_allocated_bytes_per_call"] == 0, path
                assert stats["host_allocated_bytes_per_call"] == 0, path
                assert stats["direct_gpu_allocated_bytes_per_call"] == 0, path
                assert stats["samples"] >= config["min_samples"], path
                assert all(math.isfinite(t) and t > 0 for t in stats["seconds"]), path
            data[backend, degree, precision] = result

stats = [result[operation] for result in data.values() for operation in ("analysis", "synthesis", "roundtrip")]
duration = sum(s["gpu_measured_seconds"] for s in stats)
executions = sum(s["executions"] for s in stats)
lines = ["# Transformées vectorielles CUDA — buffers préalloués", "",
    f"Batch **64**, **{duration:.3f} s** cumulées mesurées par événements CUDA, "
    f"**{executions:,} exécutions** sur 25 cas et trois opérations.", "",
    "**Zéro octet alloué sur CPU et GPU par appel chronométré**, vérifié pour les 75 opérations. "
    "Les appels directs non capturés n'allouent pas non plus de tableaux GPU ; leurs allocations CPU de pilotage sont enregistrées séparément.", "",
    "Les temps ci-dessous sont les médianes GPU d'un aller-retour complet pour les 64 vecteurs, en millisecondes.", "",
    "| T | SHTns Float64 | SHTnsKit Float32 | SpeedyTransforms Float32 | SHTnsKit Float64 | SpeedyTransforms Float64 |",
    "|---:|---:|---:|---:|---:|---:|"]
columns = [("shtns", "Float64"), ("shtnskit", "Float32"), ("speedy", "Float32"), ("shtnskit", "Float64"), ("speedy", "Float64")]
for case in config["cases"]:
    degree = case["lmax"]
    values = [f'{1000*data[b,degree,p]["roundtrip"]["median_seconds"]:.4f}' for b,p in columns]
    lines.append(f"| {degree} | " + " | ".join(values) + " |")
lines += ["", "## Échantillonnage", "",
    f"Entre **{min(s['samples'] for s in stats)} et {max(s['samples'] for s in stats)} échantillons** par opération ; "
    f"entre **{min(s['executions'] for s in stats)} et {max(s['executions'] for s in stats)} exécutions** par opération. "
    "Les répétitions internes amortissent les événements et synchronisations. La compilation, les validations et l'échauffement sont hors budget.", "",
    "| Package | Temps GPU mesuré (s) | Échantillons | Exécutions |", "|---|---:|---:|---:|"]
for backend in ("shtns", "shtnskit", "speedy"):
    subset = [r[o] for (b,_,_),r in data.items() if b == backend for o in ("analysis", "synthesis", "roundtrip")]
    lines.append(f"| {backend} | {sum(s['gpu_measured_seconds'] for s in subset):.3f} | {sum(s['samples'] for s in subset)} | {sum(s['executions'] for s in subset)} |")
lines += ["", "## Exactitude et périmètre", "",
    "Les champs analytiques communs sont validés avant les mesures. Chaque graphe est vérifié après remplissage de ses sorties avec des NaN ; les vitesses sont revérifiées après les répétitions.", "",
    "Les trois candidats utilisent la relecture de graphes CUDA complets. SHTns passe par son API publique en place. "
    "SHTnsKit réutilise ses kernels batch natifs avec buffers et plans FFT fixes ; SpeedyTransforms compose ses boucles FFT fusionnées, kernels de Legendre et opérateurs vectoriels natifs. "
    "Ces deux derniers chemins sont des adaptateurs de benchmark utilisant des API internes, identifiés dans les fichiers de résultats. Aucun algorithme mathématique n'est remplacé.", "",
    "Les fichiers TOML contiennent toutes les mesures GPU, les temps muraux, p95, compteurs d'allocations et erreurs. "
    "Ces temps ne se comparent pas directement aux anciens pipelines avec allocations et chronométrage mural.", ""]
(folder / "summary.md").write_text("\n".join(lines))
print(f"Validated: 25 cases, 75 operations, {duration:.3f} GPU seconds, {executions:,} executions, zero CPU/GPU bytes per timed call.")
