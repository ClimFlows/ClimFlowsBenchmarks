# Transformées vectorielles CUDA — buffers préalloués

Batch **64**, **60.004 s** cumulées mesurées par événements CUDA, **61,423 exécutions** sur 25 cas et trois opérations.

**Zéro octet alloué sur CPU et GPU par appel chronométré**, vérifié pour les 75 opérations. Les appels directs non capturés n'allouent pas non plus de tableaux GPU ; leurs allocations CPU de pilotage sont enregistrées séparément.

Les temps ci-dessous sont les médianes GPU d'un aller-retour complet pour les 64 vecteurs, en millisecondes.

| T | SHTns Float64 | SHTnsKit Float32 | SpeedyTransforms Float32 | SHTnsKit Float64 | SpeedyTransforms Float64 |
|---:|---:|---:|---:|---:|---:|
| 21 | 0.1864 | 0.5448 | 0.2987 | 3.1567 | 0.8064 |
| 42 | 0.6105 | 1.1324 | 0.8311 | 7.2932 | 2.5272 |
| 63 | 1.8579 | 2.2050 | 1.9236 | 14.2086 | 6.6172 |
| 127 | 9.2174 | 14.8240 | 8.8668 | 74.4439 | 28.6443 |
| 255 | 55.2086 | 191.0653 | 48.4822 | 683.9021 | 170.8110 |

## Échantillonnage

Entre **10 et 512 échantillons** par opération ; entre **10 et 9216 exécutions** par opération. Les répétitions internes amortissent les événements et synchronisations. La compilation, les validations et l'échauffement sont hors budget.

| Package | Temps GPU mesuré (s) | Échantillons | Exécutions |
|---|---:|---:|---:|
| shtns | 12.001 | 4238 | 29951 |
| shtnskit | 24.002 | 1583 | 4632 |
| speedy | 24.001 | 6200 | 26840 |

## Exactitude et périmètre

Les champs analytiques communs sont validés avant les mesures. Chaque graphe est vérifié après remplissage de ses sorties avec des NaN ; les vitesses sont revérifiées après les répétitions.

Les trois candidats utilisent la relecture de graphes CUDA complets. SHTns passe par son API publique en place. SHTnsKit réutilise ses kernels batch natifs avec buffers et plans FFT fixes ; SpeedyTransforms compose ses boucles FFT fusionnées, kernels de Legendre et opérateurs vectoriels natifs. Ces deux derniers chemins sont des adaptateurs de benchmark utilisant des API internes, identifiés dans les fichiers de résultats. Aucun algorithme mathématique n'est remplacé.

Les fichiers TOML contiennent toutes les mesures GPU, les temps muraux, p95, compteurs d'allocations et erreurs. Ces temps ne se comparent pas directement aux anciens pipelines avec allocations et chronométrage mural.
