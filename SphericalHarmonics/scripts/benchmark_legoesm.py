#!/usr/bin/env python3
"""Benchmark legoESM's Gaussian-grid vector wind/vorticity transforms."""
from __future__ import annotations

import argparse
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
import tomli_w

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    uv_from_vordiv_3d,
    vordiv_from_uv_3d,
)

ROOT = Path(__file__).resolve().parents[1]
OPS = ("analysis", "synthesis", "roundtrip")
SOURCE_REVISION = "a8ab4a655799c178a612b611687e981bc1d74de4"


def relative_error(actual, expected) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(np.float64).eps))


def block_and_check(tree, device):
    for value in jax.tree_util.tree_leaves(tree):
        if isinstance(value, jax.Array):
            value.block_until_ready()
            if device not in value.devices():
                raise RuntimeError("legoESM benchmark arrays left the selected device")
    return tree


def timed_jax(fn, args, device, warmup: int, budget: float, min_samples: int) -> dict:
    for _ in range(max(warmup, 3)):
        block_and_check(fn(*args), device)

    def sample(repeats: int) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        block_and_check(output, device)
        return (time.perf_counter() - start) / repeats

    estimate = max(sample(1), 1e-7)
    repeats = max(1, min(1000, math.ceil(0.002 / estimate)))
    seconds, spent = [], 0.0
    while spent < budget or len(seconds) < min_samples:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        block_and_check(output, device)
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


def run_case(case, batch, precision, config, options, output_dir, device):
    lmax = case["lmax"]
    chunk_size = options.chunk_size
    if chunk_size is None:
        chunk_size = 0 if lmax <= 63 else (32 if lmax <= 127 else 8)
    os.environ["LEGOESM_SH_CHUNK_SIZE"] = str(chunk_size)
    run_dtype = jnp.float32 if precision == "Float32" else jnp.float64
    coeff_dtype = jnp.complex64 if precision == "Float32" else jnp.complex128
    np_coeff_dtype = np.complex64 if precision == "Float32" else np.complex128
    grid = create_gaussian_grid(lmax, dealiasing="quadratic", run_dtype=run_dtype)
    rng = np.random.default_rng(config["seed"] + lmax)
    shape = (grid.n_sh, batch)
    vor_np = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np_coeff_dtype)
    div_np = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np_coeff_dtype)
    ls, ms = np.asarray(grid.ls), np.asarray(grid.ms)
    vor_np[ls == 0] = 0
    div_np[ls == 0] = 0
    mzero = ms == 0
    vor_np[mzero] = vor_np[mzero].real
    div_np[mzero] = div_np[mzero].real
    vor = jax.device_put(vor_np, device)
    div = jax.device_put(div_np, device)
    cos_lat = grid.cos_lat[:, None, None]

    def synthesis(vor_hat, div_hat):
        u_cos, v_cos = uv_from_vordiv_3d(grid, vor_hat, div_hat)
        return u_cos / cos_lat, v_cos / cos_lat

    def analysis(u, v):
        vor_hat, div_hat = vordiv_from_uv_3d(grid, u, v)
        # The current Float32 vector-analysis implementation promotes its
        # complex coefficients to Complex128. Include the cast needed to keep
        # a Float32/Complex64 model state in the measured Float32 path.
        if precision == "Float32":
            vor_hat, div_hat = vor_hat.astype(jnp.complex64), div_hat.astype(jnp.complex64)
        return vor_hat, div_hat

    synthesis_jit = jax.jit(synthesis)
    analysis_jit = jax.jit(analysis)
    u, v = synthesis_jit(vor, div)
    block_and_check((u, v), device)
    recovered = analysis_jit(u, v)
    block_and_check(recovered, device)
    tolerance = 5e-4 if precision == "Float32" else 5e-10
    errors = {
        "vorticity_spectral_relative_l2": relative_error(recovered[0], vor),
        "divergence_spectral_relative_l2": relative_error(recovered[1], div),
    }
    if not all(math.isfinite(value) and value <= tolerance for value in errors.values()):
        raise RuntimeError(f"T{lmax} {precision} failed vector round-trip validation: {errors}")

    result = {
        "status": "validated" if options.validate_only else "ok",
        "backend": "legoesm",
        "lmax": lmax,
        "precision": precision,
        "batch": batch,
        "nlat": grid.n_lat,
        "nlon": grid.n_lon,
        "modal_shape": list(vor.shape),
        "legoesm_sh_chunk_size": chunk_size,
        "mode": "jax-jit-device-resident",
        "errors": errors,
        "tolerance": tolerance,
        "notes": (
            "Native Gaussian-grid vector wind↔vorticity/divergence operators; quadratic dealiasing. "
            "Float32 analysis coefficients are explicitly cast from Complex128 to Complex64, and "
            "the cast is included in analysis and round-trip timing."
            if precision == "Float32" else
            "Native Gaussian-grid vector wind↔vorticity/divergence operators; quadratic dealiasing."
        ),
    }
    if not options.validate_only:
        functions = {
            "analysis": (analysis_jit, (u, v)),
            "synthesis": (synthesis_jit, (vor, div)),
            "roundtrip": (lambda x, y: synthesis_jit(*analysis_jit(x, y)), (u, v)),
        }
        case_budget = options.seconds_total / (len(options.lmax_values) * len(options.precisions) * len(OPS))
        for operation, (fn, args) in functions.items():
            stats = timed_jax(fn, args, device, options.warmup, case_budget, options.min_samples)
            stats["vector_fields_per_second"] = batch / stats["median_seconds"]
            result[operation] = stats
        after = block_and_check(functions["roundtrip"][0](u, v), device)
        result["post_measurement_velocity_error"] = max(
            relative_error(after[0], u), relative_error(after[1], v)
        )
        if result["post_measurement_velocity_error"] > tolerance:
            raise RuntimeError(f"T{lmax} {precision} failed post-measurement validation")
    write_toml(output_dir / f"T{lmax}-{precision}.toml", result)
    print(f"legoesm T{lmax} {precision} {result['status']} errors={errors}", flush=True)


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
    parser.add_argument("--chunk-size", type=int,
                        default=(int(os.environ["LEGOESM_SH_CHUNK_SIZE"])
                                 if "LEGOESM_SH_CHUNK_SIZE" in os.environ else None))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-cpu-validation", action="store_true")
    args = parser.parse_args()
    args.lmax_values = [int(v) for v in args.lmax.split(",")]
    args.precisions = args.precision.split(",")
    if args.batch <= 0 or args.seconds_total <= 0 or args.min_samples < 1:
        parser.error("batch and seconds-total must be positive; min-samples must be at least 1")
    if args.chunk_size is not None and args.chunk_size < 0:
        parser.error("chunk-size must be non-negative")
    allowed = {c["lmax"] for c in config["cases"]}
    if not args.lmax_values or any(v not in allowed for v in args.lmax_values):
        parser.error(f"lmax values must be selected from {sorted(allowed)}")
    if not args.precisions or any(p not in ("Float32", "Float64") for p in args.precisions):
        parser.error("precision must contain Float32 and/or Float64")
    if args.allow_cpu_validation and not args.validate_only:
        parser.error("--allow-cpu-validation can only be used with --validate-only")
    return args, config


def main():
    options, config = parse_args()
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if devices:
        device = devices[0]
    elif options.allow_cpu_validation:
        device = jax.devices("cpu")[0]
    else:
        raise RuntimeError("JAX CUDA is unavailable; GPU timing requires an NVIDIA GPU")
    output_dir = Path(options.out).resolve() / "legoesm"
    output_dir.mkdir(parents=True, exist_ok=True)
    env_dir = ROOT / "environments" / "legoesm"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(env_dir / name, output_dir / name)
    for name in ("benchmark_legoesm.py", "run_legoesm.sh", "setup_legoesm.sh"):
        shutil.copy2(ROOT / "scripts" / name, output_dir / name)
    shutil.copy2(ROOT / "sources" / "legoesm-core.tar.gz", output_dir / "legoesm-core.tar.gz")
    shutil.copy2(ROOT / "config.toml", output_dir / "config.toml")
    arguments = [f"--out={options.out}", f"--lmax={options.lmax}", f"--precision={options.precision}",
                f"--batch={options.batch}", f"--warmup={options.warmup}",
                f"--seconds-total={options.seconds_total}", f"--min-samples={options.min_samples}"]
    if options.validate_only:
        arguments.append("--validate-only")
    if options.allow_cpu_validation:
        arguments.append("--allow-cpu-validation")
    metadata = {
        "backend": "legoesm",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "legoesm_core_revision": SOURCE_REVISION,
        "legoesm_core_sha256": hashlib.sha256((ROOT / "sources" / "legoesm-core.tar.gz").read_bytes()).hexdigest(),
        "legoesm_chunking_policy": ("explicit" if options.chunk_size is not None
                                      else "auto: 0 through T63, 32 at T127, 8 above T127"),
        "uv_lock_sha256": hashlib.sha256((env_dir / "uv.lock").read_bytes()).hexdigest(),
        "arguments": arguments,
        "legoesm_sh_chunk_size": int(os.environ.get("LEGOESM_SH_CHUNK_SIZE", "0")),
        "measurement": "Jitted Gaussian-grid GCM vector conversions, device-resident inputs, synchronized wall time; no timed host-device transfers.",
    }
    write_toml(output_dir / "metadata.toml", metadata)
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], check=False, capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        nvidia_smi = "nvidia-smi is unavailable on this system.\n"
    (output_dir / "nvidia-smi.txt").write_text(nvidia_smi)
    write_toml(output_dir / "config.toml", config)
    selected = [c for c in config["cases"] if c["lmax"] in options.lmax_values]
    for case in sorted(selected, key=lambda c: c["lmax"], reverse=True):
        for precision in sorted(options.precisions, reverse=True):
            run_case(case, options.batch, precision, config, options, output_dir, device)


if __name__ == "__main__":
    main()
