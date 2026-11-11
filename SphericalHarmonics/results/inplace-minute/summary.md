# Vector spherical harmonic transforms

GPU median milliseconds per batch; warmed execution.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | shtnskit | Float32 | 64 | ok | 0.2529 | 0.2633 | 0.5448 |
| 21 | shtnskit | Float64 | 64 | ok | 1.6950 | 1.4421 | 3.1567 |
| 21 | speedy | Float32 | 64 | ok | 0.1158 | 0.1729 | 0.2987 |
| 21 | speedy | Float64 | 64 | ok | 0.3868 | 0.4084 | 0.8064 |
| 42 | shtnskit | Float32 | 64 | ok | 0.5954 | 0.5205 | 1.1324 |
| 42 | shtnskit | Float64 | 64 | ok | 4.4956 | 2.7720 | 7.2932 |
| 42 | speedy | Float32 | 64 | ok | 0.3681 | 0.4431 | 0.8311 |
| 42 | speedy | Float64 | 64 | ok | 1.3287 | 1.1922 | 2.5272 |
| 63 | shtnskit | Float32 | 64 | ok | 1.3164 | 0.9146 | 2.2050 |
| 63 | shtnskit | Float64 | 64 | ok | 8.1135 | 5.9165 | 14.2086 |
| 63 | speedy | Float32 | 64 | ok | 0.9678 | 0.9408 | 1.9236 |
| 63 | speedy | Float64 | 64 | ok | 3.7414 | 2.8763 | 6.6172 |
| 127 | shtnskit | Float32 | 64 | ok | 10.2420 | 4.5393 | 14.8240 |
| 127 | shtnskit | Float64 | 64 | ok | 43.1106 | 30.1526 | 74.4439 |
| 127 | speedy | Float32 | 64 | ok | 5.0838 | 3.9456 | 8.8668 |
| 127 | speedy | Float64 | 64 | ok | 17.9505 | 10.5927 | 28.6443 |
| 255 | shtnskit | Float32 | 64 | ok | 128.3319 | 38.0716 | 191.0653 |
| 255 | shtnskit | Float64 | 64 | ok | 393.4339 | 251.0884 | 683.9021 |
| 255 | speedy | Float32 | 64 | ok | 27.0887 | 18.2120 | 48.4822 |
| 255 | speedy | Float64 | 64 | ok | 112.2300 | 52.6700 | 170.8110 |

48.003 GPU seconds; 31,472 executions; 60 measured operations.
Timed Julia calls allocate zero CPU/GPU bytes per call.
