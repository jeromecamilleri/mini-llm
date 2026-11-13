"""
LEÇON 1 (sudoku_rl) : l'environnement et l'oracle
==================================================

    .venv/bin/python src/07_sudoku_rl/scripts/01_environnement.py

PLACE DANS LE PARCOURS
Ce chapitre 07 suit la même mécanique que 04_training :
  - un environnement expose un état et accepte des actions  (≈ corpus + tirer_batch)
  - un oracle retourne un signal  (≈ la cible y en supervisé)
  - une récompense remplace la perte cross-entropy comme signal d'apprentissage

Mais l'objectif est différent : maximiser la récompense cumulée (RL),
pas minimiser une perte sur un corpus fixe (supervisé).

CE QU'ON VOIT ICI
  Étape 1 : charger une grille du mini-dataset
  Étape 2 : créer l'environnement et l'oracle
  Étape 3 : jouer des coups à la main (OUI, NON)
  Étape 4 : vérifier la terminaison sur une grille pré-résolue
  Étape 5 : statistiques de l'épisode
  Étape 6 : encodage one-hot (format d'entrée du réseau, Phase 2)
"""
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from sudoku_env import SudokuEnv
from sudoku_utils import (
    afficher_grille,
    chaine_vers_grille,
    decode_action,
    encode_action,
    resoudre,
)

# ──────────────────────────────────────────────────────────────
print("=" * 55)
print("ÉTAPE 1 : charger une grille de test")
print("=" * 55)

# Grille "easy" connue pour être simple à résoudre manuellement
PUZZLE_STR   = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION_STR = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"

puzzle   = chaine_vers_grille(PUZZLE_STR)
solution = chaine_vers_grille(SOLUTION_STR)

print("Puzzle de départ :")
print(afficher_grille(puzzle))
print(f"\nCases vides : {int(np.sum(puzzle == 0))}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 2 : créer l'environnement et l'oracle")
print("=" * 55)

env = SudokuEnv(puzzle, solution, n_coups_max=200)
etat = env.reset()
print(env)
print(f"Actions légales au départ : {len(env.actions_legales())}")
print(f"  (cases vides × 9 chiffres = {env.n_cases_vides} × 9 = {env.n_cases_vides * 9})")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 3 : jouer des coups à la main")
print("=" * 55)

def jouer(env, ligne, col, chiffre, label=""):
    action = encode_action(ligne, col, chiffre)
    etat, r, done, info = env.step(action)
    print(
        f"  {label:30s} → verdict={info['oracle_verdict']:3s}  "
        f"récompense={r:+.1f}  cases_restantes={info['n_cases_restantes']}"
    )
    return done

# La case (0,1) vaut 3 dans le puzzle → déjà fixée, l'oracle dira NON
jouer(env, 0, 1, 7, "case fixée (0,1)=3, on pose 7")

# La case (0,2) vaut 0 dans le puzzle → solution[0,2] = 4
jouer(env, 0, 2, 7, "case vide (0,2), mauvais chiffre 7")
jouer(env, 0, 2, 4, "case vide (0,2), bon chiffre 4")

# La case (0,3) vaut 0 → solution[0,3] = 6 → on met 6 dans la ligne 0 (doublon ?)
# ligne 0 contient déjà 5,3,4 donc 6 est libre
jouer(env, 0, 3, 6, "case vide (0,3), bon chiffre 6")

# Doublon : on essaie de mettre 6 dans (0,4) alors qu'il y a déjà un 6 en (0,3)
jouer(env, 0, 4, 6, "doublon : 6 déjà en ligne 0")

print("\nGrille après les coups :")
print(env.render())

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 4 : terminaison sur une grille pré-résolue")
print("=" * 55)

# On crée un env avec une grille presque entièrement résolue (1 case manquante)
presque_str = SOLUTION_STR[:40] + "0" + SOLUTION_STR[41:]  # on vide la case 40
presque     = chaine_vers_grille(presque_str)
env2 = SudokuEnv(presque, solution, n_coups_max=10)

# La case manquante est à position 40 → ligne 4, col 4
# solution[4,4] vaut int(SOLUTION_STR[40])
chiffre_manquant = int(SOLUTION_STR[40])
print(f"Case manquante : ligne=4, col=4, solution={chiffre_manquant}")

action = encode_action(4, 4, chiffre_manquant)
etat, r, done, info = env2.step(action)
print(f"  verdict={info['oracle_verdict']}  récompense={r:+.1f}  terminé={done}")
print("  → la récompense +11.0 = +1.0 (coup correct) + +10.0 (bonus victoire)")
assert done, "L'épisode aurait dû se terminer !"
assert r == 11.0, f"Récompense attendue 11.0, reçu {r}"
print("  ✓ terminaison et récompense correctes")

# Test timeout
env3 = SudokuEnv(puzzle, solution, n_coups_max=3)
env3.reset()
for _ in range(3):
    _, r_t, done_t, _ = env3.step(encode_action(0, 2, 7))  # coup faux répété
print(f"\nTimeout après 3 coups : terminé={done_t}, récompense={r_t}")
assert done_t
print("  ✓ timeout correct")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 5 : statistiques de l'épisode")
print("=" * 55)

env.reset()
recompense_cumulee = 0.0
n_oui = n_non = 0

# On joue les coups corrects un par un (simuler un agent parfait)
for r_idx in range(9):
    for c_idx in range(9):
        if puzzle[r_idx, c_idx] == 0:
            bon = int(solution[r_idx, c_idx])
            action = encode_action(r_idx, c_idx, bon)
            _, rew, done, info = env.step(action)
            recompense_cumulee += rew
            if info["oracle_verdict"] == "OUI":
                n_oui += 1
            else:
                n_non += 1
            if done:
                break
    if done:
        break

print(f"  Coups joués   : {env.n_coups}")
print(f"  OUI / NON     : {n_oui} / {n_non}")
print(f"  Récompense ∑  : {recompense_cumulee:+.1f}")
print(f"  Épisode terminé : {env.termine}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 6 : encodage one-hot (entrée du réseau — Phase 2)")
print("=" * 55)

env.reset()
enc = env.etat_encode()
print(f"  Shape de l'état encodé : {enc.shape}  (9×9 cases, 10 canaux)")
print(f"  Canal 0  = cases vides : {enc[:,:,0].sum():.0f} cases")
for k in range(1, 10):
    print(f"  Canal {k}  = chiffre {k}    : {enc[:,:,k].sum():.0f} cases")
print(f"\n  Somme par case (doit être 1 partout) : min={enc.sum(axis=2).min():.0f}  max={enc.sum(axis=2).max():.0f}")
assert np.all(enc.sum(axis=2) == 1.0), "Encodage one-hot invalide !"
print("  ✓ encodage one-hot correct")

print("\n✓ Leçon 1 terminée.")
