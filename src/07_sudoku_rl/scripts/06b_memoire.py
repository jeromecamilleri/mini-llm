"""
ESSAI (leçon 6) : mémoire des coups refusés
============================================

    .venv/bin/python src/07_sudoku_rl/scripts/06b_memoire.py sudoku_reinforce_v30_lr3e3_g0 --vides 30 --horizon 60

Question : le jeu au « meilleur coup » échoue parce qu'il rejoue le même coup faux (un coup faux
ne change pas la grille). Si l'environnement retire des choix possibles un coup déjà refusé,
le glouton passe-t-il à son deuxième meilleur choix ? On ne réentraîne rien : on évalue un
checkpoint déjà entraîné, sur les mêmes 256 grilles, avec et sans mémoire.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from reinforce import evaluer_lot, grilles_presque_finies
from reseau import charger_modele

p = argparse.ArgumentParser()
p.add_argument("nom", help="checkpoint dans models/checkpoints (sans .pt)")
p.add_argument("--vides", type=int, default=30)
p.add_argument("--horizon", type=int, default=60)
p.add_argument("--appareil", default="cuda")
p.add_argument("--graine", type=int, default=0, help="graine du tirage des grilles (la même que l'entraînement)")
args = p.parse_args()

ds = SudokuDataset(RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv")
PUZ = np.stack([ds[i]["puzzle"] for i in range(len(ds))]).astype(np.int64)
SOL = np.stack([ds[i]["solution"] for i in range(len(ds))]).astype(np.int64)
rng = np.random.default_rng(args.graine)
idx = rng.integers(len(ds), size=256)
g_test = grilles_presque_finies(PUZ[idx], SOL[idx], args.vides, rng) if args.vides else PUZ[idx]
s_test = SOL[idx]

modele = charger_modele(RACINE / "models" / "checkpoints" / f"{args.nom}.pt", args.appareil)
print(f"checkpoint : {args.nom} | {args.vides} vides | horizon {args.horizon} | 256 grilles de test")
print(f"  {'mode':28s} {'résolues':>9s} {'retour':>8s} {'coups':>7s}")
for glouton in (False, True):
    for memoire in (False, True):
        m = evaluer_lot(modele, g_test, s_test, args.horizon, args.appareil, glouton=glouton, memoire=memoire)
        nom = ("meilleur coup" if glouton else "tirage") + (" + mémoire" if memoire else "")
        print(f"  {nom:28s} {m['resolues']:9.0%} {m['retour']:+8.1f} {m['coups']:7.1f}")
