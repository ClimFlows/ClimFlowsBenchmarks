# Vérification depuis un dépôt Julia vierge

Vérification tentée le 8 novembre 2026 après la campagne GPU réussie.

- Entrée : `scripts/reproduce_jll.sh`, Julia 1.11.9.
- Sources : uniquement les archives et scripts du répertoire de benchmarks.
- Les quatre archives de sources ont passé la vérification SHA-256.
- Dépôt initialement vide : `build/reproduction-jll-clean/depot`.
- `JULIA_LOAD_PATH=@:@stdlib`, aucun dépôt de secours, aucun toolkit local.
- Les paquets et plusieurs artefacts ont été téléchargés normalement par Pkg.
- Deux tentatives de `Pkg.instantiate()` dans ce dépôt ont rencontré une erreur
  de décompression d'artefact (`Data Error`). Elles ont été interrompues après
  cette erreur pendant l'attente des autres sources. Codes de sortie 130
  (lanceur) et 1 (seconde session Julia interrompue).
- Aucun fichier de l'environnement manuel ou d'un autre dépôt Julia n'a été
  ajouté à ce dépôt. Aucun benchmark GPU n'a été lancé dans cet environnement.

**La reproduction complète depuis zéro n'est donc pas encore vérifiée.**
Le benchmark réussi, utilisant les dépendances déjà installées, est décrit
dans `../summary.md`. Les échecs d'installation ne remettent pas en cause
ces mesures, mais empêchent de qualifier leur reproduction de réussie.

Les journaux `setup.log`, `setup-retry.log` et les variables du lanceur
`environment.txt` sont conservés ici. Le dossier de travail intégral reste
dans `build/reproduction-jll-clean` (non versionné).

Pour une nouvelle tentative entièrement vierge depuis la racine du dossier :

```bash
export JULIA=/chemin/vers/julia-1.11.9/bin/julia
bash scripts/reproduce_jll.sh
```

Chaque invocation crée une nouvelle destination horodatée ; le choix de
destination peut être explicité avec `SHT_BENCH_REPRO_DIR` et doit être inexistant.
