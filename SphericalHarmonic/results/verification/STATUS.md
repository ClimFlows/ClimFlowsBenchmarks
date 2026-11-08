# Vérifications réalisées le 7 novembre 2026

Julia 1.11.9, NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU, compute capability 12.0, pilote 590.48.01.

- Archives des sources : sommes SHA-256 vérifiées.
- Trois environnements instanciés ; chemins des sources dans les manifests relatifs et portables.
- Référence analytique : **24 tests réussis**, incluant des dérivées et facteurs métriques vérifiés par différences finies.
- Compilation de SHTns C 3.7.5 avec le toolkit local CUDA 13.1, pour compute capability 12.0 ; logs et identité du binaire conservés.
- Syntaxe des scripts Bash, analyse syntaxique Julia et compilation du lecteur Python vérifiées.
- **25 cas GPU réussis**, batch 64 : les cinq troncatures dans les deux précisions pour SHTnsKit et SpeedyTransforms ; les cinq troncatures en Float64 pour SHTns.
- **5 cas explicitement non pris en charge** : SHTns.jl en Float32. Aucun remplacement de précision.

Chaque cas réussi comprend la validation des deux composantes de vitesse, de la divergence et de la vorticité, puis **un échauffement et deux mesures** pour chacune des trois opérations. Les données finales sont dans `verified/`.

| Package | Précision | Maximum des erreurs relatives sur les cinq résolutions |
|---|---|---:|
| shtns | Float64 | 4.003e-12 |
| shtnskit | Float32 | 4.880e-06 |
| shtnskit | Float64 | 5.787e-12 |
| speedy | Float32 | 1.066e-05 |
| speedy | Float64 | 2.591e-12 |

La campagne standard de `config.toml` (**3 échauffements, 20 mesures**) est prête mais n’a pas été exécutée. Le tableau `verified/summary.md` contient les temps des essais de vérification ; ne pas en tirer un classement définitif.

Les fichiers de `attempts/` conservent les essais de mise au point. Ils incluent notamment un manque de mémoire alors qu’Ollama occupait environ 5,3 Gio, un conflit de versions CUDA/NVRTC, puis un problème de synchronisation du stream legacy. Ces problèmes ont été traités ; les 25 cas finaux ont tous réussi. Les appels C de préparation de SHTns utilisent désormais le stream de Julia et les chronométrages attendent tous les streams du contexte CUDA.
