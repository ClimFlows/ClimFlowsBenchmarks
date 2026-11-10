# Vector spherical harmonic transforms

GPU median milliseconds per batch; warmed execution.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | torch-harmonics | Float32 | 64 | ok | 0.0657 | 0.0727 | 0.1417 |
| 21 | torch-harmonics | Float64 | 64 | ok | 0.2612 | 0.2918 | 0.5504 |
| 42 | torch-harmonics | Float32 | 64 | ok | 0.1900 | 0.2815 | 0.4981 |
| 42 | torch-harmonics | Float64 | 64 | ok | 1.4686 | 1.3628 | 2.8600 |
| 63 | torch-harmonics | Float32 | 64 | ok | 0.4595 | 0.6917 | 1.1984 |
| 63 | torch-harmonics | Float64 | 64 | ok | 3.3056 | 4.1487 | 7.5576 |
| 127 | torch-harmonics | Float32 | 64 | ok | 4.1649 | 6.3520 | 11.8135 |
| 127 | torch-harmonics | Float64 | 64 | ok | 22.4419 | 24.7903 | 48.1967 |
| 255 | torch-harmonics | Float32 | 64 | ok | 26.4468 | 36.3655 | 70.2608 |
| 255 | torch-harmonics | Float64 | 64 | ok | 160.9065 | 170.1565 | 337.7463 |

61.656 GPU seconds; 122,237 executions; 30 measured operations.
Julia adapters use preallocated in-place buffers. torch-harmonics inputs are uploaded before capture; timed CUDA graphs contain no host↔device memcpy nodes. Device-to-device copies, where recorded, remain timed.
