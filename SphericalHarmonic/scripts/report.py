#!/usr/bin/env python3
"""Validate a recorded run and write its summary (Python >= 3.11)."""
import hashlib
import math
from pathlib import Path
import sys
import tomllib

OPERATIONS = ("analysis", "synthesis", "roundtrip")
JLL_SHA256 = "11097a94101be9941966ac91e1cd2193b78799b23bcfd71f5445d51e203349ca"


def read(path):
    return tomllib.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(folder):
    rows, measurements = [], []
    for metadata in sorted(folder.glob("*/metadata.toml")):
        backend = metadata.parent
        meta, config = read(metadata), read(backend / "config.toml")
        check(hashlib.sha256((backend / "Manifest.toml").read_bytes()).hexdigest()
              == meta["manifest_sha256"], f"{backend}: manifest checksum mismatch")
        if meta.get("shtns_binary_origin") == "jll":
            check(meta["shtns_library_sha256"] == JLL_SHA256, "Unexpected SHTns_jll binary")
        args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true")
                    for a in meta["arguments"])
        degrees = [c["lmax"] for c in config["cases"]]
        if "lmax" in args:
            degrees = [d for d in degrees if d in map(int, args["lmax"].split(","))]
        precisions = args.get("precision", ",".join(config["precisions"])).split(",")
        batch = int(args.get("batch", config["batch"]))
        minimum = int(args.get("min-samples", config["min_samples"]))
        for degree in degrees:
            for precision in precisions:
                path = backend / f"T{degree}-{precision}.toml"
                result = read(path)
                check((result["lmax"], result["precision"], result["batch"])
                      == (degree, precision, batch), f"{path}: case mismatch")
                unsupported = backend.name == "shtns" and precision == "Float32"
                expected = "unsupported" if unsupported else (
                    "validated" if "validate-only" in args else "ok")
                check(result["status"] == expected, f"{path}: {result['status']}")
                times = ["—"] * 3
                if not unsupported:
                    check(all(math.isfinite(e) and 0 <= e <= result["tolerance"]
                              for e in result["errors"].values()), f"{path}: invalid numerical errors")
                    if expected == "ok":
                        check(result["mode"] == "inplace", f"{path}: unexpected measurement mode")
                        check(result["post_measurement_velocity_error"] <= result["tolerance"],
                              f"{path}: repeated execution failed validation")
                        for operation in OPERATIONS:
                            stats = result[operation]
                            check(all(stats[key] == 0 for key in (
                                "gpu_allocated_bytes_per_call", "host_allocated_bytes_per_call",
                                "direct_gpu_allocated_bytes_per_call")), f"{path}: allocations detected")
                            check(len(stats["seconds"]) == stats["samples"] >= minimum,
                                  f"{path}: insufficient samples")
                            check(all(math.isfinite(t) and t > 0 for t in stats["seconds"]),
                                  f"{path}: invalid timings")
                            check(stats["executions"] == stats["samples"] * stats["repeats_per_sample"],
                                  f"{path}: inconsistent execution count")
                            measurements.append(stats)
                        times = [f'{1000 * result[op]["median_seconds"]:.4f}' for op in OPERATIONS]
                label = backend.name
                if label == "shtns":
                    label += " (JLL)" if meta.get("shtns_binary_origin") == "jll" else " (local)"
                rows.append((degree, label, precision, batch, expected, times))
    check(bool(rows), f"No recorded runs found in {folder}")
    lines = ["# Vector spherical harmonic transforms", "",
             "GPU median milliseconds per batch; preallocated buffers and warmed execution.", "",
             "| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |",
             "|---:|---|---|---:|---|---:|---:|---:|"]
    for degree, label, precision, batch, status, times in sorted(rows):
        lines.append(f"| {degree} | {label} | {precision} | {batch} | {status} | " + " | ".join(times) + " |")
    if measurements:
        duration = sum(s["gpu_measured_seconds"] for s in measurements)
        executions = sum(s["executions"] for s in measurements)
        lines += ["", f"{duration:.3f} GPU seconds; {executions:,} executions; "
                  f"{len(measurements)} measured operations. Zero measured CPU/GPU allocation per call."]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(rows)} cases: {folder / 'summary.md'}")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
