# 07 — Sudoku par apprentissage par renforcement

## Objectif global

Construire un réseau neuronal qui apprend à résoudre le Sudoku **par expérience**, sans qu'on lui programme les stratégies humaines. Le réseau découvre progressivement les règles du Sudoku en recevant des récompenses de l'environnement.

Ce chapitre est distinct du cours GPT (chapitres 01–06) : même mécanique de fond (gradient, optimiseur, checkpoints), architecture différente (CNN + Policy/Value), objectif différent (renforcement, pas supervision).

## Différences clés avec le GPT

| | GPT (ch. 01–06) | Sudoku RL (ch. 07) |
|---|---|---|
| Signal | Cross-entropy sur corpus fixe | Récompense de l'environnement |
| Données | Texte découpé en fenêtres | Grilles générées dynamiquement |
| Architecture | Transformer | CNN + têtes Policy/Value |
| Évaluation | Perte, perplexité | Taux de grilles résolues |

## Phases

| Phase | Objectif | État |
|---|---|---|
| **1** | Environnement, oracle, dataset | ← ici |
| 2 | Réseau CNN + boucle RL (REINFORCE / PPO) | à venir |
| 3 | Curriculum learning, amélioration | à venir |
| 4 | Export ONNX/TFLite, application mobile | à venir |

## Parcours Phase 1

| Leçon | Script | Sujet |
|---|---|---|
| 1 | `scripts/01_environnement.py` | SudokuEnv, Oracle, récompenses ([fiche](fiches/01_environnement.md)) |
| 2 | `scripts/02_dataset.py` | Chargement, solveur, difficulté ([fiche](fiches/02_dataset.md)) |
| 3 | `scripts/03_mesures.py` | Masque d'actions, agents sans apprentissage, mesures ([fiche](fiches/03_mesures.md)) |
| 4 | `scripts/04_reseau.py` | Réseau CNN, politique et valeur, masque vectorisé ([fiche](fiches/04_reseau.md)) |
| 5 | `scripts/05_env_lots.py` | Environnement par lots, vérifié contre SudokuEnv, vitesse ([fiche](fiches/05_env_lots.md)) |

Phase 1 vérifiée selon le §10 du cahier des charges (`docs/sudoku_rl_phase1_cahier_des_charges.md`), avec deux écarts volontaires : oracle binaire (pas de `PEUT_ETRE`) et 4 niveaux issus des fichiers sources, au lieu de 3 niveaux par cases vides.

## Organisation

```
tools/
  sudoku_utils.py   — encodage actions, validation, solveur backtracking, affichage
  oracle.py         — Oracle binaire (OUI / NON)
  sudoku_env.py     — SudokuEnv (interface RL standard)
  dataset.py        — SudokuDataset + construction depuis les fichiers bruts
  evaluation.py     — masque, agents sans apprentissage, mesure d'une politique
  reseau.py         — ReseauSudoku (CNN politique + valeur), encodage et masques vectorisés
  env_lots.py       — SudokuEnvLots (B grilles en tenseurs, CPU/GPU)
scripts/
  01_environnement.py
  02_dataset.py
fiches/             — 01_environnement.md, 02_dataset.md, 03_mesures.md, 04_reseau.md, 05_env_lots.md
```

## Lancement

```bash
# Leçon 1 : environnement et oracle (sans données externes)
.venv/bin/python src/07_sudoku_rl/scripts/01_environnement.py

# Leçon 2 : construction du mini-dataset (~1 min)
.venv/bin/python src/07_sudoku_rl/scripts/02_dataset.py --mini

# Leçon 2 : construction complète (~15–30 min, solveur backtracking)
.venv/bin/python src/07_sudoku_rl/scripts/02_dataset.py
```

## Données

Fichiers sources téléchargés depuis [sudoku-exchange-puzzle-bank](https://github.com/grantm/sudoku-exchange-puzzle-bank) dans `datasets/sudoku/raw/`. Domaine public.

Les fichiers CSV générés (`datasets/sudoku/`) ne sont pas versionnés (voir `.gitignore`).

## Récompenses

| Événement | Valeur |
|---|---|
| Chiffre correct (OUI) | +1.0 |
| Chiffre incorrect ou impossible (NON) | −1.0 |
| Grille entièrement résolue (bonus) | +10.0 |
| Timeout | 0.0 |
