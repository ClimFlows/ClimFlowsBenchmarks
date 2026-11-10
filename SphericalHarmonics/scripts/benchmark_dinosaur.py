#!/usr/bin/env python3
"""Benchmark Dinosaur's JAX vector spherical-harmonic conversions."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import dinosaur
from dinosaur.spherical_harmonic import (
    FastSphericalHarmonics,
    Grid,
    uv_nodal_to_vor_div_modal,
    vor_div_to_uv_nodal,
)
import tomli_w

ROOT = Path(__file__).resolve().parents[1]
OPS = ("analysis", "synthesis", "roundtrip")


def modes_for(lmax: int) -> list[tuple[int, int]]:
    return list(dict.fromkeys(((1, 0), (2, 1), (3, 2), (lmax // 2, lmax // 3), (lmax - 1, 1), (lmax, lmax))))


def legendre_mode(l: int, m: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, s = np.cos(theta), np.sin(theta)
    p = np.full_like(theta, 1.0 / math.sqrt(4 * math.pi))
    for k in range(1, m + 1):
        p *= -math.sqrt((2 * k + 1) / (2 * k)) * s
    prev = np.zeros_like(theta)
    if l > m:
        prev, p = p, math.sqrt(2 * m + 3) * x * p
        for k in range(m + 2, l + 1):
            a = math.sqrt((4 * k * k - 1) / (k * k - m * m))
            b = math.sqrt(((2 * k + 1) * ((k - 1) ** 2 - m * m)) / ((2 * k - 3) * (k * k - m * m)))
            prev, p = p, a * x * p - b * prev
    c = 0.0 if l == m else math.sqrt((2 * l + 1) / (2 * l - 1) * (l * l - m * m))
    return p, (l * x * p - c * prev) / s


def make_grid(case):
    # Dinosaur's grid axes are (longitude, latitude), matching the standard
    # T21/T42 point counts while allowing the other benchmark resolutions.
    return Grid(
        longitude_wavenumbers=case["lmax"] + 1,
        total_wavenumbers=case["lmax"] + 2,
        longitude_nodes=case["nlon"],
        latitude_nodes=case["nlat"],
        latitude_spacing="gauss",
        spherical_harmonics_impl=partial(
            FastSphericalHarmonics,
            fourier_method="fft",
            stacked_fourier_transforms=False,
        ),
        basis_as_jax_arrays=True,
    )


def fixture(grid, lmax: int, batch: int, dtype: np.dtype, seed: int, coeff_path: Path):
    modes = modes_for(lmax)
    subprocess.run([
        os.environ.get("JULIA", "julia"), "--startup-file=no",
        str(ROOT / "scripts" / "export_fixture_coefficients.jl"),
        str(lmax), str(batch), str(seed), str(coeff_path),
    ], check=True, cwd=ROOT / "scripts", stdout=subprocess.DEVNULL)
    coeff = np.fromfile(coeff_path, dtype="<f8").reshape((len(modes), batch, 4), order="F")
    longitude, sin_latitude = grid.nodal_axes
    theta = np.arccos(sin_latitude)
    phi = longitude
    nlon, nlat = grid.nodal_shape
    east = np.zeros((batch, nlon, nlat), dtype=np.float64)
    north = np.zeros_like(east)
    div = np.zeros_like(east)
    vor = np.zeros_like(east)
    invcos = 1.0 / np.cos(np.arcsin(sin_latitude))
    for q, (ell, m) in enumerate(modes):
        # Vector inversion uses a latitudinal derivative recurrence and clips
        # its aliased top-degree component, as in Dinosaur's model tendencies.
        if ell == lmax:
            continue
        ll = ell * (ell + 1)
        p, dp = legendre_mode(ell, m, theta)
        c, s = np.cos(m * phi), np.sin(m * phi)
        for b in range(batch):
            a, ai, t, ti = coeff[q, b] / ll
            chi = a * c + ai * s
            psi = t * c + ti * s
            dchi = m * (-a * s + ai * c)
            dpsi = m * (-t * s + ti * c)
            vt = dp[:, None] * chi[None, :] - p[:, None] * dpsi[None, :] * invcos[:, None]
            vp = p[:, None] * dchi[None, :] * invcos[:, None] + dp[:, None] * psi[None, :]
            # Convert colatitude/longitude components to eastward/northward winds.
            east[b] += vp.T
            north[b] -= vt.T
            div[b] -= ll * (p[:, None] * chi[None, :]).T
            vor[b] -= ll * (p[:, None] * psi[None, :]).T

    def put(value):
        return jax.device_put(value.astype(dtype, copy=True), GPU)
    return put(east), put(north), put(div), put(vor), (east, north)


def relative_error(actual, expected) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(np.float64).eps))


def block_and_check(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for value in leaves:
        if isinstance(value, jax.Array):
            value.block_until_ready()
            if GPU not in value.devices():
                raise RuntimeError("Dinosaur benchmark arrays must remain on the selected GPU")
    return tree


def timed_jax(fn, args, warmup: int, budget: float, min_samples: int) -> dict:
    for _ in range(max(warmup, 3)):
        block_and_check(fn(*args))

    def sample(repeats: int) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        block_and_check(output)
        return (time.perf_counter() - start) / repeats

    estimate = max(sample(1), 1e-7)
    repeats = max(1, min(1000, math.ceil(0.002 / estimate)))
    seconds = []
    spent = 0.0
    while spent < budget or len(seconds) < min_samples:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        block_and_check(output)
        elapsed = time.perf_counter() - start
        seconds.append(elapsed / repeats)
        spent += elapsed
    return {
        "seconds": seconds,
        "minimum_seconds": min(seconds),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        "measured_wall_seconds": spent,
        "target_wall_seconds": budget,
        "samples": len(seconds),
        "repeats_per_sample": repeats,
        "executions": len(seconds) * repeats,
        "host_to_device_transfers_timed": 0,
        "device_to_host_transfers_timed": 0,
        "inputs_resident_on_device": True,
        "outputs_resident_on_device": True,
        "execution_path": "JAX jit with device-resident inputs; synchronized wall time includes dispatch",
        "timing_scope": "synchronized wall time; includes JAX host dispatch, excludes transfers",
    }


def write_toml(path: Path, data: dict):
    path.write_text(tomli_w.dumps(data))


def run_case(case, batch, precision, config, options, output_dir):
    lmax = case["lmax"]
    dtype = np.float32 if precision == "Float32" else np.float64
    grid = make_grid(case)
    coeff_path = ROOT / "build" / "dinosaur-fixtures" / f"coefficients-T{lmax}-B{batch}.bin"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    east, north, div_field, vor_field, ref = fixture(
        grid, lmax, batch, dtype, config["seed"], coeff_path
    )

    def analysis_impl(uv):
        return uv_nodal_to_vor_div_modal(grid, uv[:, 0], uv[:, 1])

    def synthesis_impl(vor_div):
        return vor_div_to_uv_nodal(grid, vor_div[0], vor_div[1])

    analysis = jax.jit(analysis_impl)
    synthesis = jax.jit(synthesis_impl)
    velocity = jnp.stack((east, north), axis=1)
    expected = (grid.to_modal(vor_field), grid.to_modal(div_field))
    expected = block_and_check(expected)
    velocity_reference = (np.asarray(ref[0]), np.asarray(ref[1]))
    analyzed = block_and_check(analysis(velocity))
    reconstructed = block_and_check(synthesis(expected))
    errors = {
        "vorticity_spectral_relative_l2": relative_error(analyzed[0], expected[0]),
        "divergence_spectral_relative_l2": relative_error(analyzed[1], expected[1]),
        "velocity_east_relative_l2": relative_error(reconstructed[0], velocity_reference[0]),
        "velocity_north_relative_l2": relative_error(reconstructed[1], velocity_reference[1]),
    }
    tolerance = 8e-4 if dtype == np.float32 else 5e-9
    valid = all(math.isfinite(e) and e <= tolerance for e in errors.values())
    if not valid and precision == "Float64":
        result = {
            "status": "unsupported",
            "backend": "dinosaur",
            "lmax": lmax,
            "precision": precision,
            "batch": batch,
            "nlat": case["nlat"],
            "nlon": case["nlon"],
            "mode": "jax-jit-device-resident",
            "errors": errors,
            "tolerance": tolerance,
            "notes": "Dinosaur's modal basis and FFT scaling are Float32; Float64 does not meet the benchmark's 5e-9 accuracy threshold.",
        }
        write_toml(output_dir / f"T{lmax}-{precision}.toml", result)
        print(f"dinosaur T{lmax} {precision} unsupported errors={errors}", flush=True)
        return
    if not valid:
        raise RuntimeError(f"T{lmax} {precision} failed validation: {errors}; tolerance={tolerance:g}")

    result = {
        "status": "validated" if options.validate_only else "ok",
        "backend": "dinosaur",
        "lmax": lmax,
        "precision": precision,
        "batch": batch,
        "nlat": case["nlat"],
        "nlon": case["nlon"],
        "modal_shape": list(grid.modal_shape),
        "mode": "jax-jit-device-resident",
        "errors": errors,
        "tolerance": tolerance,
        "notes": "Dinosaur uses (longitude, latitude) nodal axes and a padded real modal layout; direct GCM wind↔vorticity/divergence conversions are timed.",
        "analytic_modes": [list(mode) for mode in modes_for(lmax) if mode[0] < lmax],
    }
    if not options.validate_only:
        functions = {
            "analysis": (analysis, (velocity,)),
            "synthesis": (synthesis, (expected,)),
            "roundtrip": (lambda uv: synthesis(analysis(uv)), (velocity,)),
        }
        case_budget = options.seconds_total / (len(options.lmax_values) * len(options.precisions) * len(OPS))
        for operation, (fn, args) in functions.items():
            stats = timed_jax(fn, args, options.warmup, case_budget, options.min_samples)
            stats["vector_fields_per_second"] = batch / stats["median_seconds"]
            result[operation] = stats
        after = block_and_check(functions["roundtrip"][0](velocity))
        result["post_measurement_velocity_error"] = max(
            relative_error(after[0], velocity_reference[0]),
            relative_error(after[1], velocity_reference[1]),
        )
        if result["post_measurement_velocity_error"] > tolerance:
            raise RuntimeError(f"T{lmax} {precision} failed post-measurement validation")
    write_toml(output_dir / f"T{lmax}-{precision}.toml", result)
    print(f"dinosaur T{lmax} {precision} {result['status']} errors={errors}", flush=True)


def parse_args():
    config = tomllib.loads((ROOT / "config.toml").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--lmax", default=",".join(str(c["lmax"]) for c in config["cases"]))
    parser.add_argument("--precision", default=",".join(config["precisions"]))
    parser.add_argument("--batch", type=int, default=config["batch"])
    parser.add_argument("--warmup", type=int, default=config["warmup"])
    parser.add_argument("--seconds-total", type=float, default=config["gpu_seconds_total"])
    parser.add_argument("--min-samples", type=int, default=config["min_samples"])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.lmax_values = [int(v) for v in args.lmax.split(",")]
    args.precisions = args.precision.split(",")
    if args.batch <= 0 or args.seconds_total <= 0 or args.min_samples < 1:
        parser.error("batch and seconds-total must be positive; min-samples must be at least 1")
    allowed = {c["lmax"] for c in config["cases"]}
    if not args.lmax_values or any(v not in allowed for v in args.lmax_values):
        parser.error(f"lmax values must be selected from {sorted(allowed)}")
    if not args.precisions or any(p not in ("Float32", "Float64") for p in args.precisions):
        parser.error("precision must contain Float32 and/or Float64")
    return args, config


def main():
    global GPU
    options, config = parse_args()
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX CUDA is unavailable; check the NVIDIA driver and CUDA 13 wheels")
    GPU = devices[0]
    output_dir = Path(options.out).resolve() / "dinosaur"
    output_dir.mkdir(parents=True, exist_ok=True)
    env_dir = ROOT / "environments" / "dinosaur"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(env_dir / name, output_dir / name)
    for name in ("benchmark_dinosaur.py", "export_fixture_coefficients.jl"):
        shutil.copy2(ROOT / "scripts" / name, output_dir / name)
    shutil.copy2(ROOT / "config.toml", output_dir / "config.toml")
    arguments = [f"--out={options.out}", f"--lmax={options.lmax}", f"--precision={options.precision}",
                f"--batch={options.batch}", f"--warmup={options.warmup}",
                f"--seconds-total={options.seconds_total}", f"--min-samples={options.min_samples}"]
    if options.validate_only:
        arguments.append("--validate-only")
    metadata = {
        "backend": "dinosaur",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": GPU.device_kind,
        "compute_capability": str(GPU.compute_capability),
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "dinosaur_version": dinosaur.__version__,
        "uv_lock_sha256": hashlib.sha256((env_dir / "uv.lock").read_bytes()).hexdigest(),
        "arguments": arguments,
        "measurement": "Warmed jitted GCM vector conversions with device-resident arrays; synchronized wall timing; no host-device transfers in timed region.",
    }
    write_toml(output_dir / "metadata.toml", metadata)
    (output_dir / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
    write_toml(output_dir / "config.toml", config)
    selected = [c for c in config["cases"] if c["lmax"] in options.lmax_values]
    for case in sorted(selected, key=lambda c: c["lmax"], reverse=True):
        for precision in sorted(options.precisions, reverse=True):
            run_case(case, options.batch, precision, config, options, output_dir)


if __name__ == "__main__":
    main()
