# Benchmarks Julia de transformées harmoniques sphériques sur CUDA

Périmètre convenu pour ClimFlows : **64 champs vectoriels tangents** (donc 128
composantes réelles), Float32 et Float64, T21/T42/T63/T127/T255. T511 et T1023
ne font pas partie de cette campagne. Les trois candidats sont **SHTns.jl,
SHTnsKit.jl et SpeedyTransforms.jl**.

Ce répertoire contient les sources amont figées et leurs licences, trois
environnements indépendants, les adaptateurs, une référence analytique, les
scripts de compilation et de mesure. Les dépendances transitives sont figées
par les `Manifest.toml` ; leur téléchargement initial nécessite Internet.
Les sources archivées incluent des versions de développement : les commits,
et non les numéros de version seuls, identifient les implémentations comparées.

## Campagne préallouée (protocole actuel)

### SHTns précompilé : CUDA 12.9

La campagne SHTns_jll du 8 novembre 2026 est réussie : les cinq cas Float64
T21 à T255, batch 64, sont validés ; 60,002 secondes GPU mesurées et zéro
allocation CPU/GPU par appel chronométré. Voir
[`results/jll-cuda12.9/summary.md`](results/jll-cuda12.9/summary.md).
Le binaire précompilé est SHTns_jll 3.7.0+0 (variante CUDA 12.6), utilisé avec
le runtime CUDA 12.9. La compilation locale comparée utilise SHTns 3.7.5 :
les différences de temps concernent les configurations complètes.

Pour reproduire depuis un **nouveau dépôt Julia isolé**, sans toolkit local :

```bash
export JULIA=/chemin/vers/julia-1.11.9/bin/julia
bash scripts/reproduce_jll.sh
```

Cette commande crée une nouvelle copie de travail sous `build/`, avec uniquement
les scripts, adaptateurs, sources archivées, configuration et environnement figé
du présent dossier. Elle installe les dépendances via `Pkg.instantiate()` dans
un dépôt neuf sans dépôt de secours, puis exécute validations et mesures.
Elle ne reprend ni `vendor/`, ni `build/`, ni les caches du compte utilisateur.
Les journaux, l'environnement effectif, les sources et les résultats restent
dans le répertoire créé, même en cas d'échec. Le pilote NVIDIA reste un
prérequis système. Le premier téléchargement peut représenter plusieurs Go.
La révision du wrapper corrigé est fournie par l'archive vérifiée SHA-256 ;
elle évite le défaut d'import `shtns_cfg` de la version publiée 0.4.0.

L'installation utilise le gestionnaire de paquets Julia. Les essais exploratoires
de récupération manuelle d'archives ne font pas partie de ce dépôt.

**État de vérification :** le benchmark ci-dessus est réussi. La vérification
depuis un dépôt vierge a été tentée deux fois dans un dépôt isolé et a rencontré
une erreur de décompression d'artefact ; elle n'est pas encore réussie.
Voir [`results/jll-cuda12.9/reproduction/STATUS.md`](results/jll-cuda12.9/reproduction/STATUS.md)
pour les conditions exactes et les journaux.

### Comparaison initiale des trois bibliothèques

La campagne du **8 novembre 2026** est terminée : **60,004 s GPU**,
**61 423 exécutions**, 25 cas validés, zéro allocation CPU/GPU par appel mesuré.
Le [rapport et les timings actuels](results/inplace-minute/summary.md) remplacent
les anciens essais avec allocations pour la comparaison de performances.

Le mode par défaut est désormais **`inplace`** : entrées, sorties, buffers de
travail et plans sont créés avant les mesures. Une capture du pipeline complet
est préparée puis réutilisée. Chaque cas doit avoir **zéro octet alloué sur CPU
et GPU par appel mesuré** ; sinon il est signalé en erreur. Les allocations
CPU de l'appel direct aux bibliothèques sont enregistrées séparément : elles
sont évitées dans le chemin chronométré par la relecture du graphe CUDA.

La campagne complète répartit **60 secondes mesurées par événements CUDA**
entre 25 cas et trois opérations : 12 s pour SHTns, 24 s pour SHTnsKit,
24 s pour SpeedyTransforms. Les cas coûteux sont exécutés en premier et le
budget restant est redistribué. Il y a au moins 10 échantillons par opération ;
les petits calculs sont répétés au sein d'un échantillon pour viser au moins
2 ms. Le nombre exact d'échantillons, de répétitions et d'exécutions est conservé.
Compilation, préparation, validation, capture et échauffement sont hors budget.

```bash
# Après installation et définition des variables CUDA ci-dessous :
bash scripts/run.sh all --out=results/inplace-minute
python3 scripts/report_inplace.py results/inplace-minute
```

Les temps principaux sont ceux des **événements CUDA**, par appel de batch 64.
Les temps muraux sont également conservés dans `wall_seconds` et
`wall_median_seconds`. Les événements sont préalloués. Après chaque échantillon,
tous les streams du contexte sont synchronisés. La sortie de chaque graphe est
vérifiée après remplissage des buffers de sortie avec des NaN, puis les champs
physiques sont revérifiés après la série de mesures.

Adaptations nécessaires, sans changer les algorithmes mathématiques :

* SHTns : méthodes publiques `analys!` et `synth!`, copies protectrices dans des
  buffers fixes, conversions spectrales en place.
* SHTnsKit : pas d'API vectorielle batch en place dans la version figée ;
  l'adaptateur réutilise les kernels batch natifs, avec plans FFT et buffers fixes.
* SpeedyTransforms : composition des boucles FFT fusionnées natives, des kernels
  de Legendre et des opérateurs vectoriels ; capture complète pour éviter les
  graphes FFT imbriqués et les allocations CPU du pilotage ordinaire.

Les nouveaux adaptateurs sont dans `adapters/inplace/`. Les anciens restent
accessibles avec **`--mode=allocating`** ; les résultats antérieurs dans
`results/verification/verified/` décrivent uniquement cet ancien protocole.
`--seconds-total=N` fixe le budget GPU d'un backend (avec `all`, N pour chaque
backend). `--min-samples=N` fixe le minimum d'échantillons. `--samples=N` concerne
uniquement l'ancien mode `allocating`.

## Installation

Prérequis : Linux x86_64, Bash, Git non nécessaire pour les sources archivées,
`tar`, `sha256sum`, Julia **1.11.9** (version de référence), pilote NVIDIA
compatible et GPU CUDA. Python >= 3.11 sert uniquement à produire le tableau
récapitulatif. Pour compiler SHTns : GCC/G++, make et un toolkit CUDA prenant
en charge le GPU ; les en-têtes et bibliothèques FFTW proviennent de FFTW_jll.
Aucune installation système de FFTW n'est nécessaire.

Depuis ce répertoire :

```bash
export JULIA=/chemin/vers/julia-1.11.9/bin/julia
# SHTns utilise ici le toolkit local 13.1 (voir ci-dessous).
export CUDA_PATH=/chemin/vers/cuda-13.1
# Optionnel : isoler aussi le cache de dépendances dans ce répertoire.
export JULIA_DEPOT_PATH="$PWD/.depot"
bash scripts/setup.sh
```

`setup.sh` vérifie les SHA-256 des archives et extrait `vendor/`, puis instancie
les environnements. Il ne met pas à jour les manifests existants. Pour installer
un seul candidat : `bash scripts/setup.sh shtnskit` (ou `shtns`, `speedy`).
Conserver `sources/` et les manifests avec les scripts pour reproduire une
campagne sur une autre machine. Ne pas modifier les sources dans `vendor/`.

### Bibliothèque CUDA pour SHTns.jl

SHTns_jll 3.7.0+0 propose des variantes CPU et CUDA jusqu'à 12.6. L'absence
de symboles CUDA dans un artefact CPU ne caractérise pas les variantes GPU.
Le cas `shtns-jll` utilise un environnement séparé avec les artefacts CUDA 12.6,
sans `SHTNS_PATH` ni compilation de SHTns :

```bash
bash scripts/setup.sh shtns-jll
bash scripts/run.sh shtns-jll --seconds-total=60 --out=results/jll-precompiled
```

Les résultats restent sous `results/jll-precompiled/shtns`, avec l'origine
`shtns_binary_origin = "jll"`, le chemin et le SHA-256 du binaire dans
`metadata.toml`. Les erreurs de compatibilité éventuelles sont conservées
dans les fichiers des cas ; elles ne constituent pas des timings.
Ce mode utilise l'adaptateur préalloué par défaut.

Un second environnement conserve le même binaire SHTns_jll et sélectionne
le runtime CUDA 12.9 (support de Blackwell, bibliothèques `.so.12`) :

```bash
bash scripts/setup.sh shtns-jll-cuda12.9
bash scripts/run.sh shtns-jll-cuda12.9 --seconds-total=60 --out=results/jll-cuda12.9
python3 scripts/report_shtns_jll.py results/jll-cuda12.9
```

Le rapport vérifie le SHA-256 du binaire inchangé et les allocations mesurées.
Les bibliothèques CUDA chargées sont enregistrées dans `metadata.toml`.

Pour la variante locale, le script ci-dessous construit la bibliothèque C depuis
l'archive fournie et utilise le mécanisme `SHTNS_PATH` du wrapper, sans modifier
ce dernier :

```bash
export CUDA_PATH=/chemin/vers/cuda-13.1
export CUDA_ARCH=120  # compute capability 12.0 du GPU de cette machine
bash scripts/build_shtns.sh
```

Sur cette machine, le toolkit est disponible à
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/13.1`.
Le SDK HPC range ses bibliothèques mathématiques séparément ; pour exécuter
SHTns sur cette machine, définir également :

```bash
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/math_libs/lib64
```

`run.sh` ajoute ces chemins au chargement du seul processus SHTns. Le fichier
`environments/shtns/LocalPreferences.toml` impose le runtime et le compilateur
locaux CUDA 13.1 pour éviter le mélange des bibliothèques NVRTC de la bibliothèque
C et de Julia. Conserver ce fichier avec le manifeste. Avec un autre toolkit,
adapter aussi cette préférence ; ce changement constitue une nouvelle configuration.

Sur une autre machine, adapter CUDA_PATH et CUDA_ARCH. La commande produit
`build/shtns-c/libshtns_cuda.so` et conserve les journaux de compilation.
Une bibliothèque externe peut être sélectionnée avec `SHTNS_PATH` ; archiver
alors son origine et ses options de compilation avec les résultats.
L'adaptateur vérifie que SHTns.jl charge bien le chemin demandé. Si un ancien
cache Julia a figé un autre `SHTNS_PATH`, le régénérer avant de relancer :

```bash
SHTNS_PATH="$PWD/build/shtns-c/libshtns_cuda.so" "$JULIA" --project=environments/shtns \
  -e 'using UUIDs; Base.compilecache(Base.PkgId(UUID("5dc4ea12-f621-4038-8458-a22eae100703"), "SHTns"))'
```

## Vérification et exécution

```bash
# Test de la référence analytique, sans dépendance externe ni GPU.
"$JULIA" --startup-file=no scripts/check_fixture.jl

# Vérification CUDA des trois candidats : T21, batch 64, deux précisions.
bash scripts/run.sh all --lmax=21 --validate-only --out=results/smoke

# Campagne préallouée : 5 résolutions, batch 64, environ 60 s GPU mesurées.
bash scripts/run.sh all --out=results/campaign-01

# Exemple ciblé, toujours batch 64 par défaut.
bash scripts/run.sh speedy --lmax=63,127 --precision=Float32 --seconds-total=10

# Tableau des résultats (les cas invalides et non supportés restent visibles).
python3 scripts/summarize.py results/campaign-01
```

Les commandes peuvent être lancées depuis un autre répertoire ; les chemins
des environnements et scripts sont calculés relativement à `run.sh`.
Un `--out` explicite désigne un dossier parent ; un sous-dossier par candidat
évite que les résultats s'écrasent entre eux. Utiliser un nouveau dossier par
campagne. Les autres options communes sont `--batch=N`, `--warmup=N` ; toute dérogation
à la configuration standard est enregistrée dans les arguments de la campagne.

SHTns.jl est conservé en **Float64**. Son chemin CUDA épinglé n'a pas de méthodes
Float32 utilisables par cet adaptateur : ces cas sont marqués `unsupported`,
sans conversion cachée en double précision. CUDA 5 et CUDA 6 sont séparés dans
des environnements différents pour respecter les contraintes des packages.

## Opérations et ancien protocole avec allocations

Sur une sphère de rayon 1 :

* **analysis** : vitesse tangente physique → coefficients de divergence/vorticité ;
* **synthesis** : coefficients de divergence/vorticité → vitesse tangente physique ;
* **roundtrip** : analysis puis synthesis, avec les coefficients intermédiaires
  réellement produits par analysis.

Le mode historique `--mode=allocating` mesure des **pipelines publics avec allocation des sorties** et
réutilisation des configurations/plans disponibles. Il inclut les copies
nécessaires pour préserver les entrées, les facteurs métriques et les conversions
potentiel ↔ divergence/vorticité. Il ne prétend pas mesurer des kernels seuls
ni comparer des API sans allocation. Les synchronisations internes des packages
sont conservées : elles font partie du coût d'intégration observé.

Toutes les entrées sont présentes sur le GPU avant la mesure. Aucun transfert
CPU–GPU explicite ni construction de champ analytique n'est chronométré.
Chaque mesure utilise l'horloge monotone du CPU, encadrée par des synchronisations
de **tous les streams du contexte CUDA** (`CUDA.device_synchronize`) ; cela inclut
le coût des lancements depuis Julia. Une collecte GC avant
chaque échantillon est hors chronométrage. Une opération porte sur **64 vecteurs**,
pas sur une seule composante. Les temps bruts, médiane, minimum, p95 et débit
en champs vectoriels/s sont conservés.

Spécificités :

* **SHTns.jl** : `howmany=64`, API vectorielle publique ; copies protectrices de
  l'analyse destructive incluses, stockage triangulaire compact et padding
  natif. Le signe du potentiel toroïdal est adapté à la convention commune.
  La configuration GPU est initialisée sur le stream de Julia via deux appels
  C de préparation (`cushtns_set_streams`, `cushtns_init_gpu`), pour préserver
  les dépendances entre copies, transformées et opérations spectrales. Les
  transformées elles-mêmes passent par les méthodes publiques de SHTns.jl.
* **SHTnsKit.jl** : API vectorielle par lots publique sur CuArray, stockage
  spectral rectangulaire. Les tables GPU sont préparées par les appels de
  validation avant chronométrage.
* **SpeedyTransforms.jl** : deux transformées scalaires plus opérateurs
  vectoriels natifs et facteurs cos(latitude) ; pas de boucle Julia sur les
  64 champs. Les FFT sont planifiées pour le batch 64 ; CUDA graphs activés.
  Une ligne spectrale supplémentaire sert aux dérivées vectorielles ; elle
  ne change pas le lmax des champs physiques conservés.

## Validation indépendante

`scripts/common.jl` construit des potentiels analytiques chi et psi, puis
évalue directement leurs dérivées :

```
v_theta = d_theta chi - d_phi psi / sin(theta)
v_phi   = d_phi chi / sin(theta) + d_theta psi
div     = Laplacien chi
vor     = Laplacien psi
```

Les fonctions de Legendre normalisées sont calculées par une récurrence
indépendante en Float64 ; les champs sont ensuite arrondis dans la précision
testée. Les modes incluent `(lmax,lmax)` et `(lmax-1,1)` ainsi que des modes bas
et intermédiaires. Les coefficients aléatoires sont déterministes et différents
dans chaque élément du batch. Chaque package reçoit les mêmes champs physiques,
évalués aux coordonnées exactes de sa grille gaussienne.

Quatre erreurs relatives L2 sur les échantillons physiques sont enregistrées :
les deux composantes de vitesse reconstruites, puis divergence et vorticité
reconstruites par synthèse scalaire. Cette dernière vérification détecte des
erreurs de signe, de normalisation et de facteurs métriques qu'un aller-retour
seul pourrait masquer. Seuils communs : **5e-4 en Float32, 5e-10 en Float64**.
Ils sont des critères de qualification de cette campagne, pas des bornes
d'erreur générales des bibliothèques. Le champ est parcimonieux en modes : cette
validation ne remplace pas une validation exhaustive de tous les coefficients.

Un échec produit `validation_failed` sans chronométrage ; une exception produit
`error` avec sa trace, et le script retourne un code non nul. Aucune substitution
automatique par un chemin CPU ou une autre précision n'est prévue.

## Résolutions

| Troncature | lmax | Latitude × longitude | Pas en longitude |
|---|---:|---:|---:|
| T21 | 21 | 32 × 64 | 5,625° |
| T42 | 42 | 64 × 128 | 2,8125° |
| T63 | 63 | 96 × 192 | 1,875° |
| T127 | 127 | 192 × 384 | 0,9375° |
| T255 | 255 | 384 × 768 | 0,46875° |

Grilles gaussiennes complètes quadratiques ; les latitudes ne sont pas
équidistantes. La plus petite longueur d'onde équatoriale représentée est
environ 360°/lmax, distincte du pas de grille.

## Traçabilité et limites

Chaque résultat contient les versions Julia/CUDA, le modèle GPU et sa compute
capability, un état `nvidia-smi`, le manifeste complet et son SHA-256, les commits
des sources, une copie de la configuration et des scripts Julia utilisés,
les arguments, erreurs et mesures brutes. Le temps de préparation inclut la
construction du fixture ; il ne doit pas être interprété comme un temps de
création du plan seul. Les valeurs de mémoire libre enregistrées sont des
instantanés globaux après synchronisation, **pas des pics de mémoire** ; elles
dépendent aussi du pool CUDA et des autres applications.

Pour SHTns, le SHA-256 du binaire C et les journaux de compilation sont également
copiés dans le dossier de résultats lorsqu'ils sont disponibles. La construction
Linux conserve explicitement les bibliothèques CUDA/FFTW à la liaison
(`--no-as-needed`), nécessaire avec l'ordre des arguments de la version amont.

Machine cible observée : NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU,
8151 Mio, compute capability 12.0, pilote 590.48.01. Pour comparer les temps,
conserver le même mode d'alimentation, brancher le portable sur secteur, éviter
les autres charges GPU et relever les conditions thermiques. Les benchmarks
n'imposent ni fréquences ni limite de puissance.

Voir `results/verification/STATUS.md` pour les vérifications effectivement
réalisées lors de la préparation ; distinguer ces tests de la campagne complète.
