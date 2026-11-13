"""
ÉTUDE DE CAS (leçon 6) : une divergence, l'essai 14
====================================================

    .venv/bin/python src/07_sudoku_rl/scripts/06d_divergence.py

Dans l'essai 14, le réseau qui VOIT les coups refusés s'est effondré : à partir de l'itération
≈ 300, 0 % de grilles résolues et 150 coups (le maximum) sur chaque partie, jusqu'à la fin.
Ce script décortique ce qui s'est passé, en trois temps :

    1. la courbe, chiffre par chiffre : quand l'entropie s'effondre-t-elle, par rapport à
       l'essai 10 (même réglage, sans les refus) ?
    2. le réseau final, rejoué sur une grille : que fait-il, et pourquoi ne se corrige-t-il pas ?
    3. l'ÉCHELLE DES LOGITS : le réseau de l'essai 10 contre celui de l'essai 14.

L'IDÉE À RETENIR
  La probabilité d'un coup vient d'un softmax : p = exp(score) / somme des exp(scores).
  Le gradient de REINFORCE sur le score du coup joué vaut (1 − p) × avantage.
  Si le score du coup joué dépasse les autres de plus de ≈ 20, p vaut 1,000000 en float32,
  donc (1 − p) = 0 : plus aucun gradient, quelle que soit la récompense (−1 à chaque coup).
  Le réseau est coincé pour toujours : un état « absorbant ».
"""
import csv
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

SANS, AVEC = "sudoku_reinforce_complet_lr3e3_g0", "sudoku_reinforce_complet_refus"


def lire(nom):
    with open(RACINE / "logs" / f"{nom}.csv") as f:
        return list(csv.DictReader(f))


print("=" * 70 + "\nÉTAPE 1 : la courbe (it. = itération ; entropie = hésitation du réseau)\n" + "=" * 70)
sans, avec = lire(SANS), lire(AVEC)
print(f"  {'it':>4s} | {'sans refus (essai 10)':^30s} | {'avec refus (essai 14)':^30s}")
print(f"  {'':>4s} | {'entropie':>9s} {'résolues':>9s} {'coups':>9s} | {'entropie':>9s} {'résolues':>9s} {'coups':>9s}")
for it in (1, 5, 10, 25, 30, 50, 100, 150, 200, 225, 250, 275, 282, 290, 300, 304, 350, 500):
    ligne = [f"{float(x[it - 1][k]):9.2f}" for x in (sans, avec) for k in ("entropie", "resolues", "coups")]
    print(f"  {it:4d} | {' '.join(ligne[:3])} | {' '.join(ligne[3:])}")
e = [float(x["entropie"]) for x in avec]
r = [float(x["resolues"]) for x in avec]
print(f"\n  entropie minimale de l'essai 10 : {min(float(x['entropie']) for x in sans):.3f}")
print(f"  entropie de l'essai 14 : < 0,10 dès l'it. {next(i + 1 for i, v in enumerate(e) if v < 0.10)}, "
      f"< 0,01 dès l'it. {next(i + 1 for i, v in enumerate(e) if v < 0.01)}")
print(f"  première it. avec moins de 50 % de grilles résolues : {next(i + 1 for i, v in enumerate(r) if v < 0.5)}")
print(f"  retour moyen (it. 400 à 500) : {np.mean([float(x['retour']) for x in avec[399:]]):+.1f} "
      "(150 coups tous faux = −150 ; 148 faux et 2 justes = −146)")

print("\n" + "=" * 70 + "\nÉTAPE 2 : le réseau final rejoue la grille n° 0\n" + "=" * 70)
ds = SudokuDataset(RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv")
PUZ = np.stack([ds[i]["puzzle"] for i in range(len(ds))]).astype(np.int64)
SOL = np.stack([ds[i]["solution"] for i in range(len(ds))]).astype(np.int64)
idx = np.random.default_rng(0).integers(len(ds), size=256)
chemin = RACINE / "models" / "checkpoints"
modele = charger_modele(chemin / f"{AVEC}.pt", "cuda").eval()
env = SudokuEnvLots(PUZ[idx], SOL[idx], n_coups_max=150, appareil="cuda")
torch.manual_seed(0)
actions, justes, repetitions = [], 0, 0
with torch.no_grad():
    for t in range(150):
        masques = env.masques()
        logits, _ = modele(env.etats(True), masques)
        p = F.softmax(logits, dim=1)
        a = torch.multinomial(p, 1).squeeze(1)
        deja = bool(env.refuse[0, a[0]])
        _, _, _, ok = env.step(a)
        actions.append(int(a[0]))
        justes += int(ok[0])
        repetitions += int(deja)
        if t < 4 or t == 149:
            haut = torch.topk(logits[0], 2).values
            print(f"  coup {t + 1:3d} : action {int(a[0]):3d} | p = {float(p[0].max()):.6f} | juste : {bool(ok[0])!s:5s}"
                  f" | déjà refusée : {deja!s:5s} | score du 1er − score du 2e = {float(haut[0] - haut[1]):9.0f}")
print(f"  sur 150 coups : {justes} justes, {len(set(actions))} action(s) distincte(s), {repetitions} répétitions d'un refus")
print("  → il voit le refus (le score baisse de 174 437 à 153 158), mais l'écart reste énorme : il rejoue le même coup.")

print("\n" + "=" * 70 + "\nÉTAPE 3 : l'échelle des scores (logits) sur les 256 grilles de départ\n" + "=" * 70)
for nom, etiquette in ((SANS, "essai 10 (sans refus)"), (AVEC, "essai 14 (avec refus)")):
    m = charger_modele(chemin / f"{nom}.pt", "cuda").eval()
    env = SudokuEnvLots(PUZ[idx], SOL[idx], n_coups_max=150, appareil="cuda")
    with torch.no_grad():
        masques = env.masques()
        logits, _ = m(env.etats(m.voit_refus), masques)
    v = logits[masques]
    print(f"  {etiquette:24s} : scores des coups autorisés de {float(v.min()):12.0f} à {float(v.max()):12.0f}"
          f" (écart-type {float(v.std()):10.1f})")
