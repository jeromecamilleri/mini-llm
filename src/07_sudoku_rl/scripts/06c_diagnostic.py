"""
DIAGNOSTIC (leçon 6) : où le réseau se trompe-t-il ?
=====================================================

    .venv/bin/python src/07_sudoku_rl/scripts/06c_diagnostic.py sudoku_reinforce_complet_lot256

Les essais 10 à 12 plafonnent vers 80 coups pour 50 cases vides : ≈ 30 coups sont faux. Avant de
changer encore un réglage, on regarde d'où viennent ces coups faux. On ne réentraîne rien : on
fait jouer un checkpoint au tirage, sans mémoire, sur 256 grilles, et on classe chaque coup.

Trois questions :
    1. Le coup est-il faux surtout quand la case a PLUSIEURS chiffres légaux ? Avec k chiffres
       légaux dans la case, jouer au hasard se trompe (k-1) fois sur k. Le réseau fait-il mieux ?
    2. Combien de coups faux sont des RÉPÉTITIONS d'un coup déjà refusé sur la même grille ?
       (Un coup faux ne change pas la grille : le réseau revoit le même état et rejoue pareil.)
    3. L'erreur dépend-elle de l'avancement de la partie ?
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from env_lots import SudokuEnvLots
from reseau import charger_modele

p = argparse.ArgumentParser()
p.add_argument("nom", help="checkpoint dans models/checkpoints (sans .pt)")
p.add_argument("--horizon", type=int, default=150)
p.add_argument("--appareil", default="cuda")
p.add_argument("--graine", type=int, default=0)
args = p.parse_args()

ds = SudokuDataset(RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv")
PUZ = np.stack([ds[i]["puzzle"] for i in range(len(ds))]).astype(np.int64)
SOL = np.stack([ds[i]["solution"] for i in range(len(ds))]).astype(np.int64)
idx = np.random.default_rng(args.graine).integers(len(ds), size=256)

modele = charger_modele(RACINE / "models" / "checkpoints" / f"{args.nom}.pt", args.appareil)
modele.eval()
env = SudokuEnvLots(PUZ[idx], SOL[idx], n_coups_max=args.horizon, appareil=args.appareil)
B = env.B
deja_faux = torch.zeros(B, 729, dtype=torch.bool, device=args.appareil)
n_vides_depart = env.n_vides.clone()

# une ligne par coup joué : (chiffres légaux dans la case, bon, répétition, avancement)
K, BON, REP, AVANCE = [], [], [], []
torch.manual_seed(args.graine)
with torch.no_grad():
    while not bool(env.termine.all()):
        masques = env.masques()
        logits, _ = modele(env.etats(modele.voit_refus), masques)
        a = torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1)
        actif = ~env.termine
        case = a // 9                                    # indice de la case (ligne × 9 + colonne)
        k = masques.reshape(B, 81, 9).sum(dim=2)[env._idx, case]
        rep = deja_faux[env._idx, a]
        avance = 1 - env.n_vides.float() / n_vides_depart.float()
        _, _, _, oui = env.step(a)
        deja_faux[env._idx[actif & ~oui], a[actif & ~oui]] = True
        K.append(k[actif].cpu()); BON.append(oui[actif].cpu())
        REP.append(rep[actif].cpu()); AVANCE.append(avance[actif].cpu())

K, BON, REP, AVANCE = (torch.cat(x).numpy() for x in (K, BON, REP, AVANCE))
n = len(K)
faux = ~BON
print(f"checkpoint : {args.nom} | 256 grilles | {n} coups joués dont {faux.sum()} faux ({faux.mean():.0%})")
print(f"  coups faux qui répètent un coup déjà refusé : {REP[faux].sum()} ({REP[faux].mean():.0%} des coups faux)")
print(f"  (en les retirant, il resterait {faux.sum() - REP[faux].sum()} coups faux, soit "
      f"{(faux.sum() - REP[faux].sum()) / 256:.1f} par grille)")

print("\n1. Erreur selon le nombre de chiffres légaux k dans la case jouée")
print(f"  {'k':>3s} {'coups':>7s} {'faux':>7s} {'hasard':>7s}")
for k in range(1, 10):
    m = K == k
    if m.sum() >= 30:
        print(f"  {k:3d} {m.sum():7d} {faux[m].mean():7.0%} {1 - 1 / k:7.0%}")

print("\n3. Erreur selon l'avancement (part des cases vides déjà remplies)")
print(f"  {'tranche':>10s} {'coups':>7s} {'faux':>7s} {'répétitions':>12s}")
for lo in np.arange(0, 1, 0.2):
    m = (AVANCE >= lo) & (AVANCE < lo + 0.2)
    print(f"  {lo:4.0%}-{lo + 0.2:4.0%} {m.sum():7d} {faux[m].mean():7.0%} {REP[m & faux].mean() if (m & faux).any() else 0:12.0%}")
