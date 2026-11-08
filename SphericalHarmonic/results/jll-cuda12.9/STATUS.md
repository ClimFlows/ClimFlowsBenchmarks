# CUDA 12.9 : campagne réussie le 8 novembre 2026

Les validations et les mesures T21/T42/T63/T127/T255 sont maintenant réussies
avec le binaire SHTns_jll inchangé et CUDA 12.9. Voir `summary.md`, les résultats
bruts sous `shtns/` et `run-resume.log`. Budget mesuré : 60.002 secondes GPU.
Le chemin chronométré utilise des buffers préalloués et des graphes CUDA,
sans allocation CPU/GPU mesurée ; batch de 64 champs vectoriels, Float64.
Cette campagne réutilise les artefacts installés. Elle ne constitue pas encore
une vérification depuis un dépôt vierge.

## Historique des difficultés de téléchargement

Le 8 novembre 2026, un environnement `environments/shtns-jll-cuda12.9`
a été préparé avec le manifeste de l'essai CUDA 12.6 et la préférence de
runtime `12.9`. Le binaire SHTns_jll prévu reste celui de la variante CUDA
12.6 ; le rapport exige son SHA-256 exact.

L'instanciation n'a pas abouti : téléchargement tronqué (`EOFError`) depuis
le serveur Julia, puis HTTP 504 depuis GitHub pour l'artefact CUDA_Runtime
`b842734cc2028a5166fdcefc7e5a24fb79ab3e79`. Voir `setup.log`.
La tentative suivante est restée au téléchargement et a été interrompue
(code 130) ; voir `run.log`. La vérification d'une source directe de secours
n'a pas été autorisée.

À ce stade antérieur, aucune conclusion de compatibilité ni aucun timing
n'étaient disponibles. La campagne réussie ci-dessus remplace ce constat.

Pour reprendre depuis la racine des benchmarks :

```bash
export JULIA=/chemin/vers/julia-1.11.9
bash scripts/setup.sh shtns-jll-cuda12.9
bash scripts/run.sh shtns-jll-cuda12.9 --seconds-total=60 --out=results/jll-cuda12.9
python3 scripts/report_shtns_jll.py results/jll-cuda12.9
```

Le protocole reste batch 64, Float64, T21/T42/T63/T127/T255, buffers
préalloués, au moins 10 échantillons par opération et budget de 60 secondes
GPU si l'initialisation et la validation réussissent.
