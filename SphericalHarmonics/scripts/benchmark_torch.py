#!/usr/bin/env python3
"""Benchmark torch-harmonics CUDA vector transforms."""
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
import tracemalloc

import numpy as np
import torch
import torch_harmonics as th
import torch_harmonics.fft as th_fft
import torch_harmonics.sht as th_sht
import tomli_w
from cuda.bindings import runtime as cuda_runtime

ROOT = Path(__file__).resolve().parents[1]
OPS = ("analysis", "synthesis", "roundtrip")


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().to(torch.complex128 if actual.is_complex() else torch.float64)
    b = expected.detach().to(torch.complex128 if expected.is_complex() else torch.float64)
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(torch.finfo(torch.float64).eps)).item()


def modes_for(lmax: int) -> list[tuple[int, int]]:
    return list(dict.fromkeys(((1, 0), (2, 1), (3, 2), (lmax // 2, lmax // 3), (lmax - 1, 1), (lmax, lmax))))


def install_capture_safe_irfft(nlon: int, mmax: int, dtype: torch.dtype):
    """Use an equivalent prebuilt Hermitian mask without scalar host-to-device writes."""
    mask = torch.ones(mmax, dtype=dtype, device="cuda")
    mask[0] = 0.0
    if nlon % 2 == 0 and nlon // 2 < mmax:
        mask[nlon // 2] = 0.0

    def capture_safe_irfft(x, n=None, dim=-1, **kwargs):
        transform_size = 2 * (x.size(dim) - 1) if n is None else n
        shape = [1] * x.ndim
        shape[dim] = -1
        x = torch.complex(x.real, x.imag * mask.reshape(*shape))
        return th_fft.ensure_contiguous(torch.fft.irfft(x, n=transform_size, dim=dim, **kwargs))

    th_sht.irfft = capture_safe_irfft


def inspect_cuda_graph(graph: torch.cuda.CUDAGraph) -> tuple[list[str], list[str]]:
    """Inspect graph node types and memcpy directions before timing."""
    raw = graph.raw_cuda_graph()
    status, _, count = cuda_runtime.cudaGraphGetNodes(raw, numNodes=0)
    if status != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes failed: {status}")
    status, nodes, _ = cuda_runtime.cudaGraphGetNodes(raw, numNodes=count)
    if status != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes failed: {status}")
    type_names = {
        getattr(cuda_runtime.cudaGraphNodeType, name): name.removeprefix("cudaGraphNodeType").lower()
        for name in dir(cuda_runtime.cudaGraphNodeType)
        if name.startswith("cudaGraphNodeType") and name != "cudaGraphNodeTypeCount"
    }
    node_types, kinds = [], []
    for node in nodes:
        status, node_type = cuda_runtime.cudaGraphNodeGetType(node)
        if status != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphNodeGetType failed: {status}")
        node_types.append(type_names.get(node_type, str(node_type)))
        if node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            status, params = cuda_runtime.cudaGraphMemcpyNodeGetParams(node)
            if status != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaGraphMemcpyNodeGetParams failed: {status}")
            kinds.append(str(params.kind).split(":")[0].split(".")[-1])
    return node_types, kinds


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


def fixture(nlat: int, nlon: int, lmax: int, batch: int, dtype: torch.dtype, seed: int, coeff_path: Path):
    modes = modes_for(lmax)
    subprocess.run([
        os.environ.get("JULIA", "julia"), "--startup-file=no",
        str(ROOT / "scripts" / "export_fixture_coefficients.jl"),
        str(lmax), str(batch), str(seed), str(coeff_path),
    ], check=True, cwd=ROOT / "scripts", stdout=subprocess.DEVNULL)
    coeff = np.fromfile(coeff_path, dtype="<f8").reshape((len(modes), batch, 4), order="F")
    theta = th.quadrature.precompute_latitudes(nlat, grid="legendre-gauss")[0].numpy()
    phi = 2.0 * np.pi * np.arange(nlon, dtype=np.float64) / nlon
    vt = np.zeros((nlat, nlon, batch), dtype=np.float64)
    vp = np.zeros_like(vt)
    div = np.zeros_like(vt)
    vor = np.zeros_like(vt)
    for q, (ell, m) in enumerate(modes):
        ll = ell * (ell + 1)
        p_and_d = [legendre_mode(ell, m, theta)]
        p, dp = p_and_d[0]
        c, s = np.cos(m * phi), np.sin(m * phi)
        for b in range(batch):
            a, ai, t, ti = coeff[q, b] / ll
            chi = a * c + ai * s
            psi = t * c + ti * s
            dchi = m * (-a * s + ai * c)
            dpsi = m * (-t * s + ti * c)
            invsin = 1.0 / np.sin(theta)
            vt[:, :, b] += dp[:, None] * chi[None, :] - p[:, None] * dpsi[None, :] * invsin[:, None]
            vp[:, :, b] += p[:, None] * dchi[None, :] * invsin[:, None] + dp[:, None] * psi[None, :]
            div[:, :, b] -= ll * p[:, None] * chi[None, :]
            vor[:, :, b] -= ll * p[:, None] * psi[None, :]
    def field(x):
        return torch.as_tensor(np.transpose(x, (2, 0, 1)).copy(), dtype=dtype, device="cuda")
    velocity = torch.stack((field(vt), field(vp)), dim=1)
    return velocity, field(div), field(vor), (vt, vp)


def as_result_path(root: Path, backend: str) -> Path:
    path = root / backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def timed_graph(fn, expected: torch.Tensor, warmup: int, budget: float, min_samples: int) -> dict:
    for _ in range(max(warmup, 3)):
        fn()
    torch.cuda.synchronize()

    before_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    direct = fn()
    torch.cuda.synchronize()
    direct_peak = max(0, torch.cuda.max_memory_allocated() - before_allocated)
    del direct

    reserved_before_graph = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        output = fn()
    graph.instantiate()
    node_types, memcpy_kinds = inspect_cuda_graph(graph)
    memcpy_nodes = [kind for kind in node_types if kind == "memcpy"]
    nested_nodes = [kind for kind in node_types if kind in ("graph", "conditional", "memcpybatch")]
    host_device_kinds = {"cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost", "cudaMemcpyDefault"}
    transfers = [kind for kind in memcpy_kinds if kind in host_device_kinds]
    if transfers or nested_nodes:
        raise RuntimeError(
            f"Captured graph contains CPU-GPU memcpy kinds={transfers} or "
            f"uninspected child-graph={len(nested_nodes)} nodes; "
            f"all memcpy kinds={memcpy_kinds}"
        )
    torch.cuda.synchronize()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    graph_error = relative_error(output, expected)
    if not math.isfinite(graph_error) or graph_error >= 1e-5:
        raise RuntimeError(f"CUDA graph output error {graph_error:g}")

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    current_before = torch.cuda.memory_allocated()
    tracemalloc.start()
    graph.replay()
    torch.cuda.synchronize()
    _, host_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()
    if allocated_after != current_before:
        raise RuntimeError("Live CUDA allocation changed during graph replay")

    def sample(repeats: int) -> float:
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / (1000.0 * repeats)

    estimate = max(sample(1), 1e-7)
    repeats = max(1, min(1000, math.ceil(0.002 / estimate)))
    seconds, wall_seconds = [], []
    spent = 0.0
    while spent < budget or len(seconds) < min_samples:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(repeats):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall_start
        elapsed = start.elapsed_time(stop) / 1000.0
        seconds.append(elapsed / repeats)
        wall_seconds.append(wall / repeats)
        spent += elapsed
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "minimum_seconds": min(seconds),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        "wall_median_seconds": float(np.median(wall_seconds)),
        "gpu_measured_seconds": spent,
        "target_gpu_seconds": budget,
        "samples": len(seconds),
        "repeats_per_sample": repeats,
        "executions": len(seconds) * repeats,
        "gpu_allocated_bytes_per_call": 0,
        "direct_gpu_peak_extra_bytes": direct_peak,
        "graph_pool_reserved_bytes": max(0, reserved_after - reserved_before_graph),
        "replay_tracemalloc_peak_bytes": host_peak,
        "cuda_graph_node_count": len(node_types),
        "cuda_graph_memcpy_node_count": len(memcpy_nodes),
        "cuda_graph_memcpy_kinds": memcpy_kinds,
        "execution_path": "PyTorch CUDA Graph replay with a fixed graph memory pool",
        "graph_output_relative_error": graph_error,
    }


def run_case(case, batch, dtype, precision, config, options, device_meta, output_dir):
    lmax, nlat, nlon = (case[key] for key in ("lmax", "nlat", "nlon"))
    coeff_path = ROOT / "build" / "torch-fixtures" / f"coefficients-T{lmax}-B{batch}.bin"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    velocity, div_field, vor_field, ref = fixture(
        nlat, nlon, lmax, batch, dtype, config["seed"], coeff_path
    )
    grid = "legendre-gauss"
    module_args = (nlat, nlon, lmax + 1, lmax + 1)
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        analysis_module = th.RealVectorSHT(*module_args, grid=grid).to(device="cuda").eval()
        synthesis_module = th.InverseRealVectorSHT(*module_args, grid=grid).to(device="cuda").eval()
        scalar_module = th.RealSHT(*module_args, grid=grid).to(device="cuda").eval()
    finally:
        torch.set_default_dtype(previous_default_dtype)
    ell = torch.arange(lmax + 1, device="cuda", dtype=dtype)
    laplacian = (ell * (ell + 1)).reshape(1, 1, lmax + 1, 1)
    eigen = torch.cat((-laplacian, laplacian), dim=1)
    safe_inverse = torch.zeros_like(eigen)
    safe_inverse[:, :, 1:, :] = 1.0 / eigen[:, :, 1:, :]
    # The scalar transform supplies an independent spectral reference for the
    # analytic divergence and vorticity fields.
    expected_d = scalar_module(div_field)
    expected_z = scalar_module(vor_field)
    expected_dz = torch.stack((expected_d, expected_z), dim=1)
    install_capture_safe_irfft(nlon, lmax + 1, dtype)

    def analyze(x):
        potentials = analysis_module(x)
        return potentials * eigen

    def synthesize(dz):
        return synthesis_module(dz * safe_inverse)

    analyzed = analyze(velocity)
    errors = {
        "divergence_spectral_relative_l2": relative_error(analyzed[:, 0], expected_d),
        "vorticity_spectral_relative_l2": relative_error(analyzed[:, 1], expected_z),
    }
    reconstructed = synthesize(expected_dz)
    errors["velocity_theta_relative_l2"] = relative_error(reconstructed[:, 0], torch.as_tensor(np.transpose(ref[0], (2, 0, 1)).copy(), dtype=dtype, device="cuda"))
    errors["velocity_phi_relative_l2"] = relative_error(reconstructed[:, 1], torch.as_tensor(np.transpose(ref[1], (2, 0, 1)).copy(), dtype=dtype, device="cuda"))
    tolerance = 5e-4 if dtype == torch.float32 else 5e-10
    if not all(math.isfinite(value) and value <= tolerance for value in errors.values()):
        raise RuntimeError(f"T{lmax} {precision} failed validation: {errors} (tolerance {tolerance:g})")

    result = {
        "status": "validated" if options.validate_only else "ok",
        "backend": "torch-harmonics",
        "lmax": lmax,
        "precision": precision,
        "batch": batch,
        "nlat": nlat,
        "nlon": nlon,
        "nlat_storage": nlat,
        "errors": errors,
        "tolerance": tolerance,
        "mode": "cuda-graph-replay",
        "notes": "The public API returns tensors. CUDA graph capture records those internal outputs and workspaces in a fixed graph pool; replay reuses that storage.",
        "analytic_modes": [list(mode) for mode in modes_for(lmax)],
    }
    if not options.validate_only:
        operation_functions = {
            "analysis": lambda: analyze(velocity),
            "synthesis": lambda: synthesize(expected_dz),
            "roundtrip": lambda: synthesize(analyze(velocity)),
        }
        case_budget = float(options.seconds_total) / (len(options.lmax_values) * len(options.precisions) * len(OPS))
        for operation, fn in operation_functions.items():
            expected = fn().detach().clone()
            stats = timed_graph(fn, expected, options.warmup, case_budget, options.min_samples)
            stats["vector_fields_per_second"] = batch / stats["median_seconds"]
            result[operation] = stats
        post = synthesize(analyze(velocity))
        torch.cuda.synchronize()
        post_error = max(
            relative_error(post[:, 0], torch.as_tensor(np.transpose(ref[0], (2, 0, 1)).copy(), dtype=dtype, device="cuda")),
            relative_error(post[:, 1], torch.as_tensor(np.transpose(ref[1], (2, 0, 1)).copy(), dtype=dtype, device="cuda")),
        )
        result["post_measurement_velocity_error"] = post_error
        if post_error > tolerance:
            raise RuntimeError(f"T{lmax} {precision} failed post-measurement validation: {post_error:g}")
    write_toml(output_dir / f"T{lmax}-{precision}.toml", result)
    print(f"torch-harmonics T{lmax} {precision} {result['status']} errors={errors}", flush=True)


def write_toml(path: Path, data: dict):
    path.write_text(tomli_w.dumps(data))


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
    options, config = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the PyTorch CUDA wheel and NVIDIA driver")
    output_root = Path(options.out).resolve()
    output_dir = as_result_path(output_root, "torch-harmonics")
    env_dir = ROOT / "environments" / "torch"
    for name in ("config.toml",):
        shutil.copy2(ROOT / name, output_dir / name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(env_dir / name, output_dir / name)
    for name in ("benchmark_torch.py", "export_fixture_coefficients.jl"):
        shutil.copy2(ROOT / "scripts" / name, output_dir / name)
    arguments = [
        f"--out={options.out}",
        f"--lmax={options.lmax}",
        f"--precision={options.precision}",
        f"--batch={options.batch}",
        f"--warmup={options.warmup}",
        f"--seconds-total={options.seconds_total}",
        f"--min-samples={options.min_samples}",
    ]
    if options.validate_only:
        arguments.append("--validate-only")
    device = torch.cuda.get_device_properties(0)
    lock_hash = hashlib.sha256((env_dir / "uv.lock").read_bytes()).hexdigest()
    metadata = {
        "backend": "torch-harmonics",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "total_memory_bytes": device.total_memory,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_harmonics_version": th.__version__,
        "uv_lock_sha256": lock_hash,
        "arguments": arguments,
        "measurement": "CUDA event timings over warmed graph replays; graph pool is created before timing and reused.",
    }
    write_toml(output_dir / "metadata.toml", metadata)
    (output_dir / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
    write_toml(output_dir / "config.toml", config)
    selected = [case for case in config["cases"] if case["lmax"] in options.lmax_values]
    for case in sorted(selected, key=lambda value: value["lmax"], reverse=True):
        for precision in sorted(options.precisions, reverse=True):
            dtype = torch.float32 if precision == "Float32" else torch.float64
            run_case(case, options.batch, dtype, precision, config, options, metadata, output_dir)


if __name__ == "__main__":
    main()
