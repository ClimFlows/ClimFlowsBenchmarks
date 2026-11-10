#!/usr/bin/env python3
"""Benchmark vector spherical transforms using S2FFT's JAX backend."""
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
import s2fft
from s2fft.sampling import s2_samples
from s2fft.transforms.spherical import forward_jax, inverse_jax
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


def fixture(lmax: int, bandlimit: int, batch: int, dtype: np.dtype, seed: int, coeff_path: Path):
    modes = modes_for(lmax)
    subprocess.run([
        os.environ.get("JULIA", "julia"), "--startup-file=no",
        str(ROOT / "scripts" / "export_fixture_coefficients.jl"),
        str(lmax), str(batch), str(seed), str(coeff_path),
    ], check=True, cwd=ROOT / "scripts", stdout=subprocess.DEVNULL)
    coeff = np.fromfile(coeff_path, dtype="<f8").reshape((len(modes), batch, 4), order="F")
    nlat, nlon = s2_samples.f_shape(bandlimit, sampling="gl")
    theta = s2_samples.thetas(bandlimit, sampling="gl")
    phi = 2.0 * np.pi * np.arange(nlon, dtype=np.float64) / nlon
    vt = np.zeros((nlat, nlon, batch), dtype=np.float64)
    vp = np.zeros_like(vt)
    div = np.zeros_like(vt)
    vor = np.zeros_like(vt)
    invsin = 1.0 / np.sin(theta)
    for q, (ell, m) in enumerate(modes):
        ll = ell * (ell + 1)
        p, dp = legendre_mode(ell, m, theta)
        c, s = np.cos(m * phi), np.sin(m * phi)
        for b in range(batch):
            a, ai, t, ti = coeff[q, b] / ll
            chi = a * c + ai * s
            psi = t * c + ti * s
            dchi = m * (-a * s + ai * c)
            dpsi = m * (-t * s + ti * c)
            vt[:, :, b] += dp[:, None] * chi[None, :] - p[:, None] * dpsi[None, :] * invsin[:, None]
            vp[:, :, b] += p[:, None] * dchi[None, :] * invsin[:, None] + dp[:, None] * psi[None, :]
            div[:, :, b] -= ll * p[:, None] * chi[None, :]
            vor[:, :, b] -= ll * p[:, None] * psi[None, :]
    # S2FFT's GL sampling uses L latitude nodes and 2L-1 longitude nodes.
    def on_device(x):
        return jax.device_put(np.transpose(x, (2, 0, 1)).astype(dtype, copy=True), GPU)
    velocity = jax.device_put(np.stack((np.transpose(vt, (2, 0, 1)), np.transpose(vp, (2, 0, 1))), axis=1).astype(dtype), GPU)
    return velocity, on_device(div), on_device(vor), (vt, vp), (nlat, nlon)


def relative_error(actual, expected) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(np.float64).eps))


def make_transforms(bandlimit: int, dtype):
    def forward_spin(field, spin: int):
        return jax.vmap(lambda f: forward_jax(f, bandlimit, spin=spin, sampling="gl"))(field)

    def inverse_spin(coeff, spin: int):
        return jax.vmap(lambda f: inverse_jax(f, bandlimit, spin=spin, sampling="gl"))(coeff)

    ell = jnp.arange(bandlimit, dtype=dtype)
    root_ll = jnp.sqrt(ell * (ell + 1.0))[None, :, None]
    safe_root_ll = jnp.where(root_ll == 0, 1.0, root_ll)
    inv_root_ll = jnp.where(root_ll == 0, 0.0, 1.0 / safe_root_ll)

    def analysis_impl(velocity):
        u, v = velocity[:, 0], velocity[:, 1]
        spin_plus = forward_spin(u + 1j * v, +1)
        spin_minus = forward_spin(u - 1j * v, -1)
        div = 0.5 * root_ll * (spin_plus - spin_minus)
        vor = -0.5j * root_ll * (spin_plus + spin_minus)
        return jnp.stack((div, vor), axis=1)

    def synthesis_impl(div_vor):
        div, vor = div_vor[:, 0], div_vor[:, 1]
        spin_plus = (div + 1j * vor) * inv_root_ll
        spin_minus = (-div + 1j * vor) * inv_root_ll
        q_plus = inverse_spin(spin_plus, +1)
        q_minus = inverse_spin(spin_minus, -1)
        u = jnp.real(0.5 * (q_plus + q_minus))
        v = jnp.real((q_plus - q_minus) / (2j))
        return jnp.stack((u, v), axis=1)

    return jax.jit(analysis_impl), jax.jit(synthesis_impl)


def as_result_path(root: Path) -> Path:
    path = root / "s2fft-jax"
    path.mkdir(parents=True, exist_ok=True)
    return path


def timed_jax(fn, args, warmup: int, budget: float, min_samples: int) -> dict:
    for _ in range(max(warmup, 3)):
        probe = fn(*args).block_until_ready()
        for value in jax.tree_util.tree_leaves((*args, probe)):
            if isinstance(value, jax.Array) and GPU not in value.devices():
                raise RuntimeError("JAX benchmark inputs and outputs must stay on the selected GPU")

    def sample(repeats: int) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        output.block_until_ready()
        return (time.perf_counter() - start) / repeats

    estimate = max(sample(1), 1e-7)
    repeats = max(1, min(1000, math.ceil(0.002 / estimate)))
    seconds, wall_seconds = [], []
    spent = 0.0
    while spent < budget or len(seconds) < min_samples:
        start = time.perf_counter()
        for _ in range(repeats):
            output = fn(*args)
        output.block_until_ready()
        elapsed = time.perf_counter() - start
        per_call = elapsed / repeats
        seconds.append(per_call)
        wall_seconds.append(per_call)
        spent += elapsed
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "minimum_seconds": min(seconds),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        "wall_median_seconds": float(np.median(wall_seconds)),
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
    lmax, bandlimit = case["lmax"], case["nlat"]
    dtype = np.float32 if precision == "Float32" else np.float64
    coeff_path = ROOT / "build" / "jax-fixtures" / f"coefficients-T{lmax}-B{batch}.bin"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    velocity, div_field, vor_field, ref, (nlat, nlon) = fixture(
        lmax, bandlimit, batch, dtype, config["seed"], coeff_path
    )
    jax_dtype = jnp.float32 if dtype == np.float32 else jnp.float64
    analysis, synthesis = make_transforms(bandlimit, jax_dtype)
    scalar_forward = jax.jit(jax.vmap(lambda f: forward_jax(f, bandlimit, spin=0, sampling="gl")))
    expected_d = scalar_forward(div_field)
    expected_z = scalar_forward(vor_field)
    expected_dz = jnp.stack((expected_d, expected_z), axis=1)

    analyzed = analysis(velocity)
    reconstructed = synthesis(expected_dz)
    errors = {
        "divergence_spectral_relative_l2": relative_error(analyzed[:, 0], expected_d),
        "vorticity_spectral_relative_l2": relative_error(analyzed[:, 1], expected_z),
        "velocity_theta_relative_l2": relative_error(reconstructed[:, 0], np.transpose(ref[0], (2, 0, 1))),
        "velocity_phi_relative_l2": relative_error(reconstructed[:, 1], np.transpose(ref[1], (2, 0, 1))),
    }
    tolerance = 5e-4 if dtype == np.float32 else 5e-10
    if not all(math.isfinite(value) and value <= tolerance for value in errors.values()):
        raise RuntimeError(f"T{lmax} {precision} failed validation: {errors} (tolerance {tolerance:g})")

    result = {
        "status": "validated" if options.validate_only else "ok",
        "backend": "s2fft-jax",
        "lmax": lmax,
        "precision": precision,
        "batch": batch,
        "nlat": nlat,
        "nlon": nlon,
        "nlat_storage": nlat,
        "sampling_bandlimit": bandlimit,
        "errors": errors,
        "tolerance": tolerance,
        "mode": "jax-jit-device-resident",
        "notes": "S2FFT GL sampling uses nlat=bandlimit and nlon=2*bandlimit-1; this is one fewer longitude than the existing 2*nlat grid.",
        "analytic_modes": [list(mode) for mode in modes_for(lmax)],
    }
    if not options.validate_only:
        functions = {
            "analysis": (analysis, (velocity,)),
            "synthesis": (synthesis, (expected_dz,)),
            "roundtrip": (lambda x: synthesis(analysis(x)), (velocity,)),
        }
        case_budget = options.seconds_total / (len(options.lmax_values) * len(options.precisions) * len(OPS))
        for operation, (fn, args) in functions.items():
            stats = timed_jax(fn, args, options.warmup, case_budget, options.min_samples)
            stats["vector_fields_per_second"] = batch / stats["median_seconds"]
            result[operation] = stats
        post = functions["roundtrip"][0](velocity).block_until_ready()
        result["post_measurement_velocity_error"] = max(
            relative_error(post[:, 0], np.transpose(ref[0], (2, 0, 1))),
            relative_error(post[:, 1], np.transpose(ref[1], (2, 0, 1))),
        )
        if result["post_measurement_velocity_error"] > tolerance:
            raise RuntimeError(f"T{lmax} {precision} failed post-measurement validation")
    write_toml(output_dir / f"T{lmax}-{precision}.toml", result)
    print(f"s2fft-jax T{lmax} {precision} {result['status']} errors={errors}", flush=True)


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
    args.lmax_values = [int(value) for value in args.lmax.split(",")]
    args.precisions = args.precision.split(",")
    if args.batch <= 0 or args.seconds_total <= 0 or args.min_samples < 1:
        parser.error("batch and seconds-total must be positive; min-samples must be at least 1")
    allowed = {case["lmax"] for case in config["cases"]}
    if not args.lmax_values or any(value not in allowed for value in args.lmax_values):
        parser.error(f"lmax values must be selected from {sorted(allowed)}")
    if not args.precisions or any(value not in ("Float32", "Float64") for value in args.precisions):
        parser.error("precision must contain Float32 and/or Float64")
    return args, config


def main():
    global GPU
    options, config = parse_args()
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX CUDA is unavailable; check JAX CUDA 13 wheels and the NVIDIA driver")
    GPU = devices[0]
    output_dir = as_result_path(Path(options.out).resolve())
    env_dir = ROOT / "environments" / "jax"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(env_dir / name, output_dir / name)
    for name in ("benchmark_jax.py", "export_fixture_coefficients.jl"):
        shutil.copy2(ROOT / "scripts" / name, output_dir / name)
    shutil.copy2(ROOT / "config.toml", output_dir / "config.toml")
    arguments = [f"--out={options.out}", f"--lmax={options.lmax}", f"--precision={options.precision}",
                f"--batch={options.batch}", f"--warmup={options.warmup}",
                f"--seconds-total={options.seconds_total}", f"--min-samples={options.min_samples}"]
    if options.validate_only:
        arguments.append("--validate-only")
    metadata = {
        "backend": "s2fft-jax",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": GPU.device_kind,
        "compute_capability": str(GPU.compute_capability),
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "s2fft_version": "1.5.0",
        "uv_lock_sha256": hashlib.sha256((env_dir / "uv.lock").read_bytes()).hexdigest(),
        "arguments": arguments,
        "measurement": "Warmed jitted calls with device-resident arrays; synchronized wall-clock timing; no host materialization or host-device transfers in timed region.",
    }
    write_toml(output_dir / "metadata.toml", metadata)
    (output_dir / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
    write_toml(output_dir / "config.toml", config)
    selected = [case for case in config["cases"] if case["lmax"] in options.lmax_values]
    for case in sorted(selected, key=lambda value: value["lmax"], reverse=True):
        for precision in sorted(options.precisions, reverse=True):
            run_case(case, options.batch, precision, config, options, output_dir)


if __name__ == "__main__":
    main()
