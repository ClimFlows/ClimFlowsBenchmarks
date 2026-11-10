# Vector spherical harmonic transforms

Synchronized JAX wall median milliseconds per batch; host dispatch included.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | s2fft-jax | Float32 | 64 | ok | 7.3982 | 6.5473 | 11.6383 |
| 21 | s2fft-jax | Float64 | 64 | ok | 7.9584 | 6.7401 | 13.5997 |
| 42 | s2fft-jax | Float32 | 64 | ok | 34.8534 | 35.3192 | 64.9138 |
| 42 | s2fft-jax | Float64 | 64 | ok | 31.5052 | 35.5337 | 62.8944 |
| 63 | s2fft-jax | Float32 | 64 | ok | 76.9256 | 78.3035 | 153.8857 |
| 63 | s2fft-jax | Float64 | 64 | ok | 70.8216 | 78.4684 | 152.6852 |
| 127 | s2fft-jax | Float32 | 64 | ok | 792.0500 | 819.1495 | 1623.8123 |
| 127 | s2fft-jax | Float64 | 64 | ok | 769.6631 | 802.2036 | 1614.1350 |

76.051 synchronized wall seconds; 482 executions; 24 measured operations.
JAX backends use device-resident arrays; timed calls contain no host↔device transfers. Reported JAX times are synchronized wall times and include host dispatch.
