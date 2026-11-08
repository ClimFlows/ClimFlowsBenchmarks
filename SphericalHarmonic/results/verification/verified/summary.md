# Vérification des pipelines — deux échantillons seulement

Ces temps servent à vérifier la reproductibilité des scripts. Ils ne constituent pas une campagne de comparaison statistique.

| Backend | T | Precision | Batch | Status | Analysis ms | Synthesis ms | Roundtrip ms | Max rel. error |
|---|---:|---|---:|---|---:|---:|---:|---:|
| shtns | 21 | Float32 | 64 | unsupported | — | — | — | — |
| shtns | 21 | Float64 | 64 | ok | 0.520 | 0.519 | 1.760 | 7.61e-15 |
| shtns | 42 | Float32 | 64 | unsupported | — | — | — | — |
| shtns | 42 | Float64 | 64 | ok | 1.096 | 0.712 | 1.336 | 6.49e-14 |
| shtns | 63 | Float32 | 64 | unsupported | — | — | — | — |
| shtns | 63 | Float64 | 64 | ok | 1.447 | 1.243 | 2.406 | 6.94e-14 |
| shtns | 127 | Float32 | 64 | unsupported | — | — | — | — |
| shtns | 127 | Float64 | 64 | ok | 5.252 | 4.385 | 9.397 | 3.82e-13 |
| shtns | 255 | Float32 | 64 | unsupported | — | — | — | — |
| shtns | 255 | Float64 | 64 | ok | 31.459 | 25.062 | 55.660 | 4.00e-12 |
| shtnskit | 21 | Float32 | 64 | ok | 1.131 | 1.238 | 2.523 | 3.82e-07 |
| shtnskit | 21 | Float64 | 64 | ok | 2.440 | 2.433 | 4.283 | 9.32e-15 |
| shtnskit | 42 | Float32 | 64 | ok | 1.484 | 1.723 | 2.615 | 6.86e-07 |
| shtnskit | 42 | Float64 | 64 | ok | 6.636 | 4.028 | 8.516 | 9.23e-14 |
| shtnskit | 63 | Float32 | 64 | ok | 2.231 | 2.237 | 4.185 | 1.15e-06 |
| shtnskit | 63 | Float64 | 64 | ok | 9.255 | 7.281 | 15.856 | 1.67e-13 |
| shtnskit | 127 | Float32 | 64 | ok | 11.049 | 6.621 | 17.236 | 2.31e-06 |
| shtnskit | 127 | Float64 | 64 | ok | 42.912 | 31.495 | 74.698 | 2.03e-13 |
| shtnskit | 255 | Float32 | 64 | ok | 103.403 | 40.022 | 124.154 | 4.88e-06 |
| shtnskit | 255 | Float64 | 64 | ok | 333.118 | 236.596 | 556.209 | 5.79e-12 |
| speedy | 21 | Float32 | 64 | ok | 1.126 | 2.953 | 1.316 | 4.96e-07 |
| speedy | 21 | Float64 | 64 | ok | 1.402 | 4.106 | 3.028 | 1.80e-14 |
| speedy | 42 | Float32 | 64 | ok | 1.912 | 2.585 | 3.110 | 1.07e-06 |
| speedy | 42 | Float64 | 64 | ok | 5.751 | 5.021 | 12.015 | 4.67e-14 |
| speedy | 63 | Float32 | 64 | ok | 3.899 | 5.740 | 17.178 | 1.71e-06 |
| speedy | 63 | Float64 | 64 | ok | 23.516 | 15.179 | 36.633 | 7.89e-14 |
| speedy | 127 | Float32 | 64 | ok | 16.435 | 15.363 | 43.587 | 4.23e-06 |
| speedy | 127 | Float64 | 64 | ok | 76.282 | 68.792 | 133.535 | 5.30e-13 |
| speedy | 255 | Float32 | 64 | ok | 130.418 | 114.821 | 254.556 | 1.07e-05 |
| speedy | 255 | Float64 | 64 | ok | 330.221 | 287.880 | 640.525 | 2.59e-12 |
