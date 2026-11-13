"""
LEÇON 3 (sudoku_rl) : mesurer avant d'apprendre
================================================

    .venv/bin/python src/07_sudoku_rl/scripts/03_mesures.py          # 200 grilles par niveau
    .venv/bin/python src/07_sudoku_rl/scripts/03_mesures.py --test   # 20 grilles, très rapide

PLACE DANS LE PARCOURS
Avant d'entraîner un réseau, on fixe la règle de mesure et on joue des agents
SANS apprentissage pour connaître le point de départ et le plafond. C'est la
même idée que la leçon 15 de 06_corpus (une mesure fiable avant de comparer).

DÉFINITIONS
  politique   : une fonction qui choisit l'action à partir de l'état. Plus tard : le réseau.
  masque      : liste des actions autorisées (case vide, pas de doublon). Il ne dépend
                que des règles, pas de la solution : le réseau pourra l'utiliser.
  épisode     : une partie, de la grille de départ à la victoire ou au timeout.
  retour      : somme des récompenses d'un épisode.

LES QUATRE AGENTS
  1. hasard                : une action parmi les 729, sans règle.
  2. hasard + masque       : une action parmi les actions autorisées.
  3. case la plus contrainte : la case vide qui a le moins de chiffres autorisés,
                             puis un chiffre autorisé au hasard (méthode humaine simple).
  4. agent parfait         : calculé, pas joué : une case = un coup juste.

CE QU'ON VOIT ICI
  Étape 1 : charger le dataset et vérifier le masque
  Étape 2 : mesurer les trois agents sans apprentissage, par niveau
  Étape 3 : lire les résultats et choisir la bonne mesure

CE QU'ON DÉCOUVRE
  Un coup faux ne modifie pas la grille (l'oracle répond NON). Avec le masque,
  le hasard finit donc toujours par résoudre : le taux de grilles résolues ne
  permet plus de comparer. La bonne mesure est le NOMBRE DE COUPS (ou le retour)
  pour finir : le minimum est le nombre de cases vides, le hasard + masque en
  demande environ le double.
"""
import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from evaluation import (
    evaluer,
    masque_actions,
    politique_case_contrainte,
    politique_hasard,
    politique_hasard_masque,
)
from sudoku_utils import a_doublon

CSV_MINI = RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv"
N_GRILLES = 20 if "--test" in sys.argv else 200
N_COUPS_MAX = 500
NIVEAUX = ["facile", "moyen", "difficile", "diabolique"]

# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 : charger le dataset et vérifier le masque")
print("=" * 60)

if not CSV_MINI.exists():
    sys.exit("Mini-dataset absent : lancer d'abord 02_dataset.py --mini")
ds = SudokuDataset(CSV_MINI)
print(f"  {ds}")

grille = ds[0]["puzzle"]
masque = masque_actions(grille).reshape(9, 9, 9)
ecarts = sum(
    (grille[r, c] == 0 and not a_doublon(grille, r, c, d)) != masque[r, c, d - 1]
    for r in range(9) for c in range(9) for d in range(1, 10)
)
print(f"  actions possibles           : 729")
print(f"  actions autorisées (masque) : {int(masque.sum())}")
print(f"  écarts masque / règles      : {ecarts}   (doit être 0)")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"ÉTAPE 2 : trois agents sans apprentissage ({N_GRILLES} grilles par niveau)")
print("=" * 60)

AGENTS = [
    ("hasard", politique_hasard),
    ("hasard + masque", politique_hasard_masque),
    ("case contrainte", politique_case_contrainte),
]
print(f"  n_coups_max = {N_COUPS_MAX}\n")
print(f"  {'niveau':11s} {'agent':16s} {'résolues':>9s} {'retour':>9s} {'coups ok':>9s} {'coups':>7s}  durée")

resume = {}
for niveau in NIVEAUX:
    d = ds.par_difficulte(niveau)
    n_vides = np.mean([(d[i]["puzzle"] == 0).sum() for i in range(N_GRILLES)])
    for nom, f in AGENTS:
        t0 = time.time()
        r = evaluer(lambda rng, f=f: (lambda env: f(env, rng)), d, N_GRILLES, N_COUPS_MAX)
        resume[(niveau, nom)] = (r, n_vides)
        coups = "-" if np.isnan(r["coups_si_resolue"]) else f"{r['coups_si_resolue']:.0f}"
        print(
            f"  {niveau:11s} {nom:16s} {r['taux_resolu']:9.0%} {r['retour_moyen']:+9.1f} "
            f"{r['part_oui']:9.0%} {coups:>7s}  {time.time() - t0:.1f} s"
        )
    parfait = 10 + n_vides  # (n−1) × 1 + 11 = n + 10
    print(f"  {niveau:11s} {'parfait (calculé)':16s} {1:9.0%} {parfait:+9.1f} {1:9.0%} {n_vides:7.0f}")
    print()

# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 3 : lecture")
print("=" * 60)
print("""  • 'hasard' (sans masque) : jamais de victoire, retour très négatif.
    Les 729 actions contiennent surtout des actions impossibles.
  • 'hasard + masque' : toutes les grilles sont résolues, mais en environ 1,7 fois
    plus de coups que le minimum, car environ 4 coups sur 10 sont faux.
  • 'case contrainte' : la méthode humaine de base se rapproche du minimum
    (92 à 97 % de coups justes), même sur les grilles « diabolique ».
    C'est ce que le réseau peut apprendre à découvrir tout seul.

  Mesure retenue pour la suite : le retour moyen et le nombre de coups, avec
  n_coups_max fixé (le taux de résolution ne distingue plus rien dès qu'on masque).
  Jalons : hasard + masque (départ) → case contrainte (méthode humaine) → parfait.""")
print("\n✓ Leçon 3 terminée.")
