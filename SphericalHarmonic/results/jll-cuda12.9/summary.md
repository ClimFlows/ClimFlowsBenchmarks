# SHTns_jll avec CUDA 12.9.0

Binaire SHTns_jll 3.7.0+0 inchangé (variante CUDA 12.6), runtime CUDA 12.9.0, batch 64.

| Troncature | Statut Float64 | Aller-retour GPU (ms) | Compilation locale CUDA 13.1 (ms) |
|---|---|---:|---:|
| T21 | ok | 0.1910 | 0.1864 |
| T42 | ok | 0.6627 | 0.6105 |
| T63 | ok | 2.0300 | 1.8579 |
| T127 | ok | 10.1473 | 9.2174 |
| T255 | ok | 60.7018 | 55.2086 |

Calcul GPU mesuré : 60.002 s ; 144568 exécutions ; 17381 échantillons.

Temps médians par batch de 64 champs vectoriels ; buffers préalloués et relecture de graphes CUDA.
Les références locales proviennent de la campagne antérieure, sans exécution simultanée.
Float32 non pris en charge par l'API CUDA du wrapper Julia retenu.

## Bibliothèques chargées

- `/lib/x86_64-linux-gnu/libcuda.so.1`
- `/home/dubos/.julia/artifacts/b842734cc2028a5166fdcefc7e5a24fb79ab3e79/lib/libcudart.so`
- `/home/dubos/.julia/artifacts/b842734cc2028a5166fdcefc7e5a24fb79ab3e79/lib/libcufft.so`
- `/home/dubos/.julia/artifacts/b842734cc2028a5166fdcefc7e5a24fb79ab3e79/lib/libnvJitLink.so.12`
- `/home/dubos/.julia/artifacts/b842734cc2028a5166fdcefc7e5a24fb79ab3e79/lib/libnvrtc.so`
- `/home/dubos/.julia/artifacts/b842734cc2028a5166fdcefc7e5a24fb79ab3e79/lib/libnvrtc-builtins.so`
