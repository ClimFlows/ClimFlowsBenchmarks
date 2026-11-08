# SHTns_jll : essai des binaires précompilés

Le benchmark a été exécuté le 8 novembre 2026 sur la NVIDIA RTX PRO 1000
Blackwell Generation Laptop GPU (compute capability 12.0), batch de 64 champs
vectoriels, buffers préalloués, budget demandé de 60 secondes GPU.

**Aucun timing valide : les cinq cas Float64 échouent à l'initialisation GPU.**
La minute de calcul n'a donc pas été consommée. Les cas Float32 restent non
pris en charge par l'API CUDA du wrapper Julia retenu.

| Troncature | Float64 | Float32 |
|---|---|---|
| T21 | Échec d'initialisation | API non prise en charge |
| T42 | Échec d'initialisation | API non prise en charge |
| T63 | Échec d'initialisation | API non prise en charge |
| T127 | Échec d'initialisation | API non prise en charge |
| T255 | Échec d'initialisation | API non prise en charge |

Le journal montre que VkFFT appelle NVRTC avec une architecture GPU que
NVRTC 12.6 ne reconnaît pas : `NVRTC_ERROR_INVALID_OPTION`,
`invalid value for --gpu-architecture (-arch)`, puis
`vkfft init FAILED with error code 4031`. Ce résultat concerne ce binaire et
ce GPU Blackwell ; il ne démontre pas une incompatibilité sur des GPU antérieurs.

## Binaire effectivement chargé

- SHTns_jll **3.7.0+0**, variante `x86_64-linux-gnu-cuda+12.6`.
- Artefact : `ffcc41cbe2007581425c4e27fb1c3ee331143ec0`.
- SHA-256 de `lib/libshtns.so` :
  `11097a94101be9941966ac91e1cd2193b78799b23bcfd71f5445d51e203349ca`.
- Runtime CUDA 12.6 fourni par les artefacts Julia ; aucun `SHTNS_PATH`.
- Aucune compilation locale de SHTns pour cet essai.

La provenance est vérifiée par l'adaptateur et enregistrée dans
`shtns/metadata.toml`. `SHTns_jll-Artifacts.toml` contient l'URL et le
SHA-256 de l'archive distribuée. Les erreurs par cas sont dans `shtns/T*.toml`
et le diagnostic natif complet dans `run.log`.

## Reproduction depuis la racine du répertoire de benchmarks

```bash
export JULIA=/chemin/vers/julia-1.11.9
bash scripts/setup.sh shtns-jll
bash scripts/run.sh shtns-jll --seconds-total=60 --out=results/jll-precompiled
```

L'environnement et ses préférences sont dans `environments/shtns-jll`.
Le lanceur retire `SHTNS_PATH`, même si cette variable existe dans le shell.
Le mode `all` conserve les trois candidats de la campagne initiale ; ce cas
précompilé se lance explicitement avec `shtns-jll`.

La campagne réussie avec SHTns compilé localement sous CUDA 13.1 reste dans
`../inplace-minute/summary.md`. Elle est distincte de cet essai précompilé.

## Même binaire avec le toolkit local CUDA 13.1

Un second essai de chargement du binaire inchangé, avec uniquement les
répertoires CUDA locaux dans `LD_LIBRARY_PATH`, échoue avant l'initialisation :
`libnvrtc.so.12: cannot open shared object file`.
Les entrées ELF `DT_NEEDED` demandent `libnvrtc.so.12` et `libcudart.so.12` ;
le toolkit local fournit `libnvrtc.so.13` et `libcudart.so.13`.
Ajouter son répertoire au chemin de recherche ne suffit donc pas.
Le diagnostic et les dépendances complètes sont dans `local-cuda-13.1.log`.

```bash
python3 scripts/probe_jll_local.py /chemin/artefact/lib/libshtns.so /chemin/cuda-13.1
```

Ce test ne modifie ni le binaire, ni ses dépendances ELF, ni les bibliothèques
locales. Il ne teste pas une substitution forcée des bibliothèques CUDA 12
par CUDA 13 sous des noms artificiels. Un toolkit CUDA 12 plus récent, avec
NVRTC prenant en charge Blackwell, serait un autre candidat à examiner.
