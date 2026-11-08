# Vector spherical harmonic transforms

GPU median milliseconds per batch; preallocated buffers and warmed execution.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | shtns (JLL) | Float32 | 64 | unsupported | — | — | — |
| 21 | shtns (JLL) | Float64 | 64 | ok | 0.1053 | 0.0869 | 0.1910 |
| 42 | shtns (JLL) | Float32 | 64 | unsupported | — | — | — |
| 42 | shtns (JLL) | Float64 | 64 | ok | 0.3538 | 0.2941 | 0.6627 |
| 63 | shtns (JLL) | Float32 | 64 | unsupported | — | — | — |
| 63 | shtns (JLL) | Float64 | 64 | ok | 1.1252 | 0.9065 | 2.0300 |
| 127 | shtns (JLL) | Float32 | 64 | unsupported | — | — | — |
| 127 | shtns (JLL) | Float64 | 64 | ok | 5.5800 | 4.5372 | 10.1473 |
| 255 | shtns (JLL) | Float32 | 64 | unsupported | — | — | — |
| 255 | shtns (JLL) | Float64 | 64 | ok | 32.4052 | 27.6633 | 60.7018 |

60.002 GPU seconds; 144,568 executions; 15 measured operations. Zero measured CPU/GPU allocation per call.
