# Vector spherical harmonic transforms

Synchronized JAX wall median milliseconds per batch; host dispatch included.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | s2fft-jax | Float32 | 64 | ok | 15.8722 | 14.0953 | 28.6921 |
| 21 | s2fft-jax | Float64 | 64 | ok | 16.0716 | 13.1081 | 26.2926 |
| 42 | s2fft-jax | Float32 | 64 | ok | 69.7704 | 75.8133 | 147.0821 |
| 42 | s2fft-jax | Float64 | 64 | ok | 71.3442 | 74.6251 | 148.9312 |
| 63 | s2fft-jax | Float32 | 64 | ok | 256.3216 | 275.7037 | 539.1711 |
| 63 | s2fft-jax | Float64 | 64 | ok | 261.5667 | 277.8972 | 545.1316 |
| 127 | s2fft-jax | Float32 | 64 | ok | 2486.7713 | 2656.8501 | 5146.7624 |
| 127 | s2fft-jax | Float64 | 64 | ok | 2514.4149 | 2659.7190 | 5200.6554 |

239.740 synchronized wall seconds; 468 executions; 24 measured operations.
S2FFT uses device-resident JAX arrays; timed calls contain no host↔device transfers. Reported JAX times are synchronized wall times and include host dispatch.
