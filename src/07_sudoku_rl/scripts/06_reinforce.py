"""
LEÇON 6 (sudoku_rl) : REINFORCE, le réseau apprend en jouant
=============================================================

    # test court (quelques minutes, CPU)
    .venv/bin/python src/07_sudoku_rl/scripts/06_reinforce.py --iterations 30 --vides 5 --appareil cpu
    # entraînement (GPU), lancé par toi
    .venv/bin/python src/07_sudoku_rl/scripts/06_reinforce.py --iterations 2000 --vides 5 --appareil cuda

PLACE DANS LE PARCOURS
Jusqu'ici le réseau joue au hasard parmi les coups autorisés (leçon 4). On lui apprend
maintenant à préférer les bons coups, avec la règle la plus simple du renforcement.

L'IDÉE : REINFORCE
  1. Le réseau joue des parties. Chaque coup reçoit une récompense (+1 bon, −1 faux, +10 victoire).
  2. Pour chaque coup, le RETOUR G = récompenses futures cumulées (les plus lointaines
     comptent un peu moins : facteur GAMMA = 0,99 par coup).
  3. AVANTAGE = G comparé à la moyenne de tous les coups du lot, divisé par l'écart-type.
     Positif : ce coup a mené à mieux que d'habitude. Négatif : à moins bien.
  4. perte = − log π(coup) × avantage. Descendre cette perte augmente la probabilité
     des coups à avantage positif et diminue celle des autres.
  Comme pour le GPT : forward, perte, backward, pas d'Adam. Seule la perte change :
  elle ne compare plus à une bonne réponse connue, elle pondère par ce qui s'est passé.

LES RÉGLAGES
  --vides K : on ne laisse que K cases vides par grille (le reste est rempli avec la
              solution). Parties courtes = signal clair. 0 = grilles complètes (≈ 50 vides).
  --lot B   : nombre de parties jouées en parallèle par itération.
  --horizon : nombre de coups maximum par partie (timeout).
  Le masque d'actions est gardé (variable déjà fixée en leçon 3) ; PEUT_ETRE reste à décider
  d'après ces essais.

CE QU'ON VOIT ICI
  Étape 1 : mesure du réseau avant apprentissage (le hasard masqué)
  Étape 2 : entraînement, avec vitesse mesurée (jeu / apprentissage) et courbe
  Étape 3 : mesure après, sur les MÊMES grilles de test, et sauvegarde
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from env_lots import SudokuEnvLots
from reinforce import evaluer_lot, grilles_presque_finies, jouer_lot, pas_reinforce, retours
from reseau import ReseauSudoku

p = argparse.ArgumentParser()
p.add_argument("--iterations", type=int, default=30)
p.add_argument("--lot", type=int, default=64)
p.add_argument("--horizon", type=int, default=0, help="0 = vides + 20 (ou 150 si grilles complètes)")
p.add_argument("--vides", type=int, default=5)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--lr-fin", type=float, default=None, help="si donné, lr décroît linéairement de --lr à cette valeur")
p.add_argument("--gamma", type=float, default=0.99, help="0 = chaque coup jugé sur sa seule récompense")
p.add_argument("--appareil", default="cpu")
p.add_argument("--graine", type=int, default=0)
p.add_argument("--complet", action="store_true", help="dataset de 200 000 grilles au lieu du mini")
p.add_argument("--nom", default="sudoku_reinforce")
p.add_argument("--refus", action="store_true", help="le réseau voit les coups déjà refusés (9 canaux d'entrée en plus)")
p.add_argument("--entropie", type=float, default=0.0, help="poids du bonus d'entropie (0 = aucun)")
args = p.parse_args()

horizon = args.horizon or (args.vides + 20 if args.vides else 150)
chemin = RACINE / "datasets" / "sudoku" / ("sudoku_dataset.csv" if args.complet else "sudoku_dataset_mini.csv")
if not chemin.exists():
    sys.exit(f"Dataset absent : {chemin}")
ds = SudokuDataset(chemin)
PUZ = np.stack([ds[i]["puzzle"] for i in range(len(ds))]).astype(np.int64)
SOL = np.stack([ds[i]["solution"] for i in range(len(ds))]).astype(np.int64)
rng = np.random.default_rng(args.graine)
torch.manual_seed(args.graine)
app = args.appareil


def tirer(n):
    idx = rng.integers(len(ds), size=n)
    g, s = PUZ[idx], SOL[idx]
    return (grilles_presque_finies(g, s, args.vides, rng) if args.vides else g), s


# grilles de test fixes, pour comparer avant et après
g_test, s_test = tirer(256)
modele = ReseauSudoku(voit_refus=args.refus).to(app)
optim = torch.optim.Adam(modele.parameters(), lr=args.lr)
print(f"appareil : {app} | vides : {args.vides or 'complètes'} | horizon : {horizon} | lot : {args.lot}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nÉTAPE 1 : avant apprentissage\n" + "=" * 60)
avant = evaluer_lot(modele, g_test, s_test, horizon, app)
minimum = args.vides or 50
print(f"  résolues {avant['resolues']:.0%} | retour {avant['retour']:+.1f} | coups {avant['coups']:.1f}"
      f"  (minimum possible : {minimum} coups)")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nÉTAPE 2 : entraînement\n" + "=" * 60)
journal = RACINE / "logs" / f"{args.nom}.csv"
t_jeu = t_app = 0.0
env = None
with open(journal, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iteration", "retour", "resolues", "coups", "perte", "entropie"])
    for it in range(1, args.iterations + 1):
        if args.lr_fin is not None:
            lr = args.lr + (args.lr_fin - args.lr) * (it - 1) / max(1, args.iterations - 1)
            for groupe in optim.param_groups:
                groupe["lr"] = lr
        g, s = tirer(args.lot)
        if env is None:
            env = SudokuEnvLots(g, s, n_coups_max=horizon, appareil=app)
        else:
            env.charger(g, s)
        t0 = time.time()
        gr, ac, re, actifs, refus = jouer_lot(env, modele, horizon)
        G = retours(re, args.gamma)
        if app == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        perte, entropie = pas_reinforce(modele, optim, gr, ac, G, actifs, refus, bonus_entropie=args.entropie)
        if app == "cuda":
            torch.cuda.synchronize()
        t_jeu += t1 - t0
        t_app += time.time() - t1
        retour = float(re.sum(0).mean())
        resolues = float((env.n_vides == 0).float().mean())
        coups = float(env.n_coups.float().mean())
        w.writerow([it, f"{retour:.3f}", f"{resolues:.3f}", f"{coups:.2f}", f"{perte:.4f}", f"{entropie:.3f}"])
        f.flush()  # le notebook peut suivre l'essai en direct
        if it % max(1, args.iterations // 20) == 0 or it == 1:
            print(f"  it {it:5d} | retour {retour:+7.2f} | résolues {resolues:4.0%} | coups {coups:6.1f}"
                  f" | entropie {entropie:.3f}")
total = t_jeu + t_app
print(f"\n  temps : jeu {t_jeu:.1f} s ({t_jeu / total:.0%}), apprentissage {t_app:.1f} s ({t_app / total:.0%})"
      f" → {args.iterations / total:.2f} itérations/s")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nÉTAPE 3 : après apprentissage (mêmes grilles de test)\n" + "=" * 60)
apres = evaluer_lot(modele, g_test, s_test, horizon, app)
glout = evaluer_lot(modele, g_test, s_test, horizon, app, glouton=True)
for nom, m in [("avant", avant), ("après (tirage)", apres), ("après (meilleur coup)", glout)]:
    print(f"  {nom:22s}: résolues {m['resolues']:4.0%} | retour {m['retour']:+7.2f} | coups {m['coups']:6.1f}")
dossier = RACINE / "models" / "checkpoints"
dossier.mkdir(parents=True, exist_ok=True)
torch.save(modele.state_dict(), dossier / f"{args.nom}.pt")
print(f"\n  courbe : {journal.relative_to(RACINE)} | modèle : models/checkpoints/{args.nom}.pt")
