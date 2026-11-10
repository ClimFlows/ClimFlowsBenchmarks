# Vector spherical harmonic transforms

GPU median milliseconds per batch; warmed execution.

| T | Backend | Precision | Batch | Status | Analysis | Synthesis | Round trip |
|---:|---|---|---:|---|---:|---:|---:|
| 21 | shtns (JLL) | Float32 | 64 | ok | 0.0228 | 0.0167 | 0.0393 |
| 21 | shtns (JLL) | Float64 | 64 | ok | 0.1037 | 0.0859 | 0.1905 |
| 42 | shtns (JLL) | Float32 | 64 | ok | 0.0576 | 0.0417 | 0.0973 |
| 42 | shtns (JLL) | Float64 | 64 | ok | 0.3407 | 0.2858 | 0.6405 |
| 63 | shtns (JLL) | Float32 | 64 | ok | 0.1741 | 0.1136 | 0.3030 |
| 63 | shtns (JLL) | Float64 | 64 | ok | 1.0825 | 0.8634 | 1.9421 |
| 127 | shtns (JLL) | Float32 | 64 | ok | 1.1247 | 0.6543 | 1.7485 |
| 127 | shtns (JLL) | Float64 | 64 | ok | 5.3215 | 4.3512 | 9.6982 |
| 255 | shtns (JLL) | Float32 | 64 | ok | 9.7061 | 6.8628 | 16.5308 |
| 255 | shtns (JLL) | Float64 | 64 | ok | 31.4682 | 26.7869 | 58.6924 |

60.000 GPU seconds; 449,018 executions; 30 measured operations.
Timed Julia calls allocate zero CPU/GPU bytes per call.
