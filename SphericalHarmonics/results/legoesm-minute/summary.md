# Vector spherical harmonic transforms

Synchronized JAX wall median milliseconds per batch; host dispatch included.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | legoesm | Float32 | 64 | ok | 0.1263 | 0.3802 | 0.4960 |
| 21 | legoesm | Float64 | 64 | ok | 0.8244 | 0.6339 | 1.3855 |
| 42 | legoesm | Float32 | 64 | ok | 0.5476 | 2.2365 | 2.8099 |
| 42 | legoesm | Float64 | 64 | ok | 2.9603 | 3.3122 | 5.9903 |
| 63 | legoesm | Float32 | 64 | ok | 1.6939 | 6.8462 | 8.6795 |
| 63 | legoesm | Float64 | 64 | ok | 8.0343 | 9.5418 | 17.2206 |
| 127 | legoesm | Float32 | 64 | ok | 13.0602 | 51.5008 | 65.8970 |
| 127 | legoesm | Float64 | 64 | ok | 50.5800 | 65.9000 | 122.6439 |
| 255 | legoesm | Float32 | 64 | ok | 107.1156 | 442.3808 | 562.2467 |
| 255 | legoesm | Float64 | 64 | ok | 391.6139 | 676.6048 | 1119.7519 |

82.340 synchronized wall seconds; 39,908 executions; 30 measured operations.
JAX backends use device-resident arrays; timed calls contain no host↔device transfers. Reported JAX times are synchronized wall times and include host dispatch.
