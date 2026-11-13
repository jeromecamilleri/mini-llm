"""
LEÇON 5 (sudoku_rl) : un environnement par lots
================================================

    .venv/bin/python src/07_sudoku_rl/scripts/05_env_lots.py

PLACE DANS LE PARCOURS
SudokuEnv (leçon 1) joue UNE grille, lisible mais lente quand le réseau veut
256 décisions à la fois. Ici : SudokuEnvLots, les mêmes règles pour B grilles
dans des tenseurs, sans boucle Python. SudokuEnv reste la référence : on vérifie
que les deux donnent exactement les mêmes résultats.

STRUCTURE DE DONNÉES
  grille (9, 9)                      le chiffre posé dans chaque case
  ligne_a, col_a, bloc_a (9, 9)      « le chiffre d est-il déjà dans cette ligne / colonne / ce bloc ? »
  Poser un chiffre met à jour 3 booléens. Le masque des actions autorisées se déduit
  de ces trois tableaux, sans relire toute la grille.

Une grille terminée est « gelée » : elle ignore les actions suivantes et rapporte 0.
TIMEOUT
  Ce n'est pas un temps en secondes : c'est un nombre de coups maximum (n_coups_max).
  Au dernier coup autorisé, si la grille n'est pas finie, la partie est coupée et ce
  coup rapporte 0. L'étape 2 utilise une limite basse (70) pour voir des timeouts ET
  des victoires : sinon la règle du timeout ne serait jamais testée.

CE QU'ON VOIT ICI
  Étape 1 : le masque de l'environnement par lots = celui de la leçon 4
  Étape 2 : mêmes récompenses, fins de partie et grilles que SudokuEnv, coup par coup
  Étape 3 : vitesse, ancien montage contre lots (CPU, GPU)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from env_lots import SudokuEnvLots
from evaluation import masque_actions
from reseau import masques_actions
from sudoku_env import SudokuEnv

CSV_MINI = RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv"
if not CSV_MINI.exists():
    sys.exit("Mini-dataset absent : lancer d'abord 02_dataset.py --mini")
ds = SudokuDataset(CSV_MINI)
B = 256
puzzles = np.stack([ds[i]["puzzle"] for i in range(B)]).astype(np.int64)
solutions = np.stack([ds[i]["solution"] for i in range(B)]).astype(np.int64)


def tirer_masque(masques: torch.Tensor, generateur) -> torch.Tensor:
    """Une action au hasard parmi les autorisées (action 0 si aucune)."""
    m = masques.clone()
    m[m.sum(dim=1) == 0, 0] = True
    return torch.multinomial(m.float(), 1, generator=generateur).squeeze(1)


# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 : le masque")
print("=" * 60)
lots = SudokuEnvLots(puzzles, solutions)
ref = masques_actions(torch.as_tensor(puzzles))
print(f"  masques identiques à la leçon 4 : {bool((lots.masques() == ref).all())}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 : mêmes résultats que SudokuEnv, coup par coup")
print("=" * 60)
N_COUPS_MAX = 70  # assez petit pour que certaines parties finissent par timeout
lots = SudokuEnvLots(puzzles, solutions, n_coups_max=N_COUPS_MAX)
refs = [SudokuEnv(p.astype(np.int8), s.astype(np.int8), n_coups_max=N_COUPS_MAX)
        for p, s in zip(puzzles, solutions)]
rng = np.random.default_rng(0)
gen = torch.Generator().manual_seed(0)
ecarts = {"récompense": 0, "terminé": 0, "grille": 0, "masque": 0}
n_compares = n_victoires = n_timeouts = 0
for coup in range(N_COUPS_MAX):
    # trois types de coups pour couvrir tous les cas : n'importe lequel, autorisé, bon chiffre
    actions = rng.integers(729, size=B)
    autorise = tirer_masque(lots.masques(), gen).numpy()
    type_coup = rng.integers(3, size=B)
    for i in range(B):
        if type_coup[i] == 1:
            actions[i] = autorise[i]
        elif type_coup[i] == 2:
            vides = np.argwhere(lots.grille[i].numpy() == 0)
            if len(vides):
                l, c = vides[rng.integers(len(vides))]
                actions[i] = l * 81 + c * 9 + int(solutions[i, l, c]) - 1
    g, r, fini, _ = lots.step(torch.as_tensor(actions))
    for i in range(B):
        if refs[i].termine:
            continue
        g_ref, r_ref, f_ref, _ = refs[i].step(int(actions[i]))
        n_compares += 1
        ecarts["récompense"] += abs(float(r[i]) - r_ref) > 1e-6
        ecarts["terminé"] += bool(fini[i]) != f_ref
        ecarts["grille"] += not np.array_equal(g[i].numpy(), g_ref)
        ecarts["masque"] += not np.array_equal(
            lots.masques()[i].numpy(), masque_actions(g_ref.astype(np.int64)))
        if f_ref:
            n_victoires += refs[i].n_cases_vides == 0
            n_timeouts += refs[i].n_cases_vides > 0
print(f"  coups comparés : {n_compares}  (grilles terminées : {n_victoires} victoires, {n_timeouts} timeouts)")
for k, v in ecarts.items():
    print(f"  écarts {k:10s} : {v}   (doit être 0)")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 : vitesse (parties jouées au hasard parmi les coups autorisés)")
print("=" * 60)
COUPS = 100

# ancien montage : une grille à la fois
t0 = time.time()
n = 0
rng2 = np.random.default_rng(1)
for i in range(B):
    e = SudokuEnv(puzzles[i].astype(np.int8), solutions[i].astype(np.int8), n_coups_max=COUPS)
    for _ in range(COUPS):
        autorisees = np.flatnonzero(masque_actions(e.grille))
        if len(autorisees) == 0 or e.termine:
            break
        e.step(int(rng2.choice(autorisees)))
        n += 1
dt = time.time() - t0
print(f"  SudokuEnv + masque en boucle : {n / dt:10.0f} coups/s")

for app in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    env = SudokuEnvLots(puzzles, solutions, n_coups_max=COUPS, appareil=app)
    g = torch.Generator(device=app).manual_seed(2)
    for _ in range(3):  # échauffement
        env.step(tirer_masque(env.masques(), g))
    env.reset()
    if app == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for _ in range(COUPS):
        _, _, fini, _ = env.step(tirer_masque(env.masques(), g))
        n += int((~fini).sum())
        if bool(fini.all()):
            break
    if app == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  SudokuEnvLots sur {app:5s}       : {n / dt:10.0f} coups/s  (lot de {B})")

print("\n✓ Leçon 5 terminée.")
