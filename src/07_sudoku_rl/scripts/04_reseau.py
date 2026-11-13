"""
LEÇON 4 (sudoku_rl) : le réseau CNN, politique et valeur
=========================================================

    .venv/bin/python src/07_sudoku_rl/scripts/04_reseau.py

PLACE DANS LE PARCOURS
En 04_training, le réseau (GPT) lisait des tokens et prédisait le suivant.
Ici le réseau lit une grille et dit QUELLE ACTION jouer. Même mécanique
(nn.Module, forward), autre architecture : des convolutions.

DÉFINITIONS
  convolution  : un petit filtre (ici 3×3) glissé sur la grille ; les mêmes poids
                 servent à chaque position (la règle du Sudoku est la même partout).
  canal        : une « carte » 9×9 de valeurs. L'entrée en a 10 (vide, 1…9),
                 les couches cachées en ont 64.
  politique    : pour chaque action, un score ; softmax → probabilités. C'est elle qui joue.
  valeur       : un nombre par grille, « ce que j'espère encore gagner d'ici la fin ».
                 Inutile pour jouer ; servira à apprendre (leçons 05 et suivantes).
  masque       : actions interdites (case remplie ou doublon) mises à probabilité 0.
  résiduel     : sortie = entrée + correction ; aide les réseaux plus profonds à apprendre.

CE QU'ON VOIT ICI
  Étape 1 : encodage one-hot vectorisé (comparé à celui de l'environnement)
  Étape 2 : masques vectorisés (comparés à ceux de la leçon 3)
  Étape 3 : le réseau, ses formes et ses paramètres
  Étape 4 : réseau non entraîné = hasard + masque (même mesure que la leçon 3)
  Étape 5 : vitesse d'un lot de grilles

CE QU'ON DÉCOUVRE
  Dernière couche de la politique initialisée à zéro : tous les scores sont égaux,
  la politique masquée est exactement uniforme, donc le réseau de départ joue comme
  l'agent « hasard + masque » (environ 90 coups). L'entraînement partira de ce
  repère connu et devra descendre vers environ 52.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset
from evaluation import evaluer, masque_actions
from reseau import ReseauSudoku, encoder_etats, masques_actions, politique_reseau
from sudoku_env import SudokuEnv

CSV_MINI = RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv"
if not CSV_MINI.exists():
    sys.exit("Mini-dataset absent : lancer d'abord 02_dataset.py --mini")
ds = SudokuDataset(CSV_MINI)
torch.manual_seed(0)

# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 : encodage one-hot vectorisé")
print("=" * 60)
grilles = np.stack([ds[i]["puzzle"] for i in range(256)])
env = SudokuEnv(ds[0]["puzzle"], ds[0]["solution"])
ancien = env.etat_encode().transpose(2, 0, 1)
nouveau = encoder_etats(grilles[:1])[0].numpy()
print(f"  forme d'un lot            : {tuple(encoder_etats(grilles).shape)}  (B, canaux, 9, 9)")
print(f"  identique à l'environnement : {np.array_equal(ancien, nouveau)}")
t = time.time()
for _ in range(200):
    encoder_etats(grilles)
print(f"  256 grilles encodées en   : {(time.time() - t) / 200 * 1e3:.2f} ms")
t = time.time()
for g in grilles:
    SudokuEnv(g, g).etat_encode()
print(f"  même travail, une par une avec l'environnement : {(time.time() - t) * 1e3:.0f} ms")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 : masques vectorisés")
print("=" * 60)
m_lot = masques_actions(torch.as_tensor(grilles, dtype=torch.long)).numpy()
ecarts = sum(
    int(not np.array_equal(m_lot[i], masque_actions(grilles[i]))) for i in range(len(grilles))
)
print(f"  grilles dont le masque diffère de la leçon 3 : {ecarts} sur {len(grilles)}  (doit être 0)")
print(f"  actions autorisées par grille : moyenne {m_lot.sum(axis=1).mean():.0f} sur 729")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 : le réseau")
print("=" * 60)
modele = ReseauSudoku(canaux=64, n_blocs=4)
print(f"  {modele.nb_parametres():,} paramètres".replace(",", " "))
x = encoder_etats(grilles[:8])
masque = masques_actions(torch.as_tensor(grilles[:8], dtype=torch.long))
with torch.no_grad():
    logits, valeur = modele(x, masque)
print(f"  entrée   : {tuple(x.shape)}")
print(f"  logits   : {tuple(logits.shape)}   (729 actions)")
print(f"  valeur   : {tuple(valeur.shape)}   (une par grille)")
probas = F.softmax(logits, dim=1)
print(f"  proba sur actions interdites : {float(probas[~masque].max()):.1e}   (doit être 0)")
n_auto = masque.sum(dim=1).float()
print(f"  proba maximale d'une action  : {float(probas.max(dim=1).values.mean()):.4f}"
      f"  (1 / autorisées = {float((1 / n_auto).mean()):.4f} : politique uniforme)")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 : réseau non entraîné = hasard + masque")
print("=" * 60)
d = ds.par_difficulte("facile")
r = evaluer(lambda rng: (lambda e: politique_reseau(modele)(e, rng)), d, 50, 500)
print(f"  50 grilles faciles : résolues {r['taux_resolu']:.0%}, retour {r['retour_moyen']:+.1f}, "
      f"coups {r['coups_si_resolue']:.0f}")
print("  (leçon 3, hasard + masque : retour +25, environ 87 coups)")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 : vitesse d'un lot")
print("=" * 60)
appareils = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for app in appareils:
    m = ReseauSudoku().to(app)
    for B in (64, 256, 1024):
        g = torch.as_tensor(np.resize(grilles, (B, 9, 9)), dtype=torch.long)
        def un_lot():
            lg, v = m(encoder_etats(g).to(app), masques_actions(g).to(app))
            if app == "cuda":
                torch.cuda.synchronize()
        un_lot()
        t = time.time()
        for _ in range(5):
            un_lot()
        dt = (time.time() - t) / 5
        print(f"  {app:5s} lot {B:5d} : {dt * 1e3:7.1f} ms  → {B / dt:9.0f} décisions/s")
print("\n✓ Leçon 4 terminée.")
