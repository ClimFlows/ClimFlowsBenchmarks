# Vector spherical harmonic transforms

GPU median milliseconds per batch; warmed execution.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | shtns (JLL) | Float32 | 64 | ok | 0.0228 | 0.0166 | 0.0393 |
| 21 | shtns (JLL) | Float64 | 64 | ok | 0.1036 | 0.0868 | 0.1906 |
| 42 | shtns (JLL) | Float32 | 64 | ok | 0.0561 | 0.0430 | 0.0983 |
| 42 | shtns (JLL) | Float64 | 64 | ok | 0.3406 | 0.2858 | 0.6396 |
| 63 | shtns (JLL) | Float32 | 64 | ok | 0.1734 | 0.1149 | 0.3114 |
| 63 | shtns (JLL) | Float64 | 64 | ok | 1.0806 | 0.8705 | 1.9496 |
| 127 | shtns (JLL) | Float32 | 64 | ok | 1.1439 | 0.6516 | 1.7288 |
| 127 | shtns (JLL) | Float64 | 64 | ok | 5.3353 | 4.3434 | 9.6886 |
| 255 | shtns (JLL) | Float32 | 64 | ok | 9.6064 | 6.6956 | 16.3282 |
| 255 | shtns (JLL) | Float64 | 64 | ok | 30.8405 | 26.3062 | 57.4194 |

60.001 GPU seconds; 448,571 executions; 30 measured operations.
Timed Julia calls allocate zero CPU/GPU bytes per call.
