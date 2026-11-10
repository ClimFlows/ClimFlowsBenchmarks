# Vector spherical harmonic transforms

Synchronized JAX wall median milliseconds per batch; host dispatch included.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | dinosaur | Float32 | 64 | ok | 0.2137 | 0.2026 | 0.3612 |
| 21 | dinosaur | Float64 | 64 | unsupported | — | — | — |
| 42 | dinosaur | Float32 | 64 | ok | 0.8135 | 1.0090 | 1.8578 |
| 42 | dinosaur | Float64 | 64 | unsupported | — | — | — |
| 63 | dinosaur | Float32 | 64 | ok | 2.7005 | 2.9088 | 5.3988 |
| 63 | dinosaur | Float64 | 64 | unsupported | — | — | — |
| 127 | dinosaur | Float32 | 64 | ok | 14.8797 | 16.4762 | 31.8966 |
| 127 | dinosaur | Float64 | 64 | unsupported | — | — | — |
| 255 | dinosaur | Float32 | 64 | ok | 95.6134 | 100.1694 | 201.1077 |
| 255 | dinosaur | Float64 | 64 | unsupported | — | — | — |

30.077 synchronized wall seconds; 29,338 executions; 15 measured operations.
JAX backends use device-resident arrays; timed calls contain no host↔device transfers. Reported JAX times are synchronized wall times and include host dispatch.
