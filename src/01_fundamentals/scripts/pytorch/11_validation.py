"""
LEÇON 11 : entraînement et validation
=====================================

    .venv/bin/python src/01_fundamentals/scripts/pytorch/11_validation.py

PLACE DANS LE PARCOURS (partie PyTorch de 01_fundamentals)
Même réseau [2, 4, 1] (tanh, sigmoid), même XOR. Mais jusqu'ici, on mesurait
la perte sur les 4 MÊMES exemples que ceux de l'entraînement : impossible de
savoir si le réseau a COMPRIS le XOR ou s'il a juste appris ses 4 réponses
par cœur. Ici on teste le réseau sur des points qu'il n'a JAMAIS vus.

LE XOR « EN CONTINU »
Les entrées ne sont plus seulement 0 ou 1 mais n'importe quel point (x1, x2)
du carré [0, 1]². La réponse vraie reste le XOR : 1 si UNE SEULE des deux
coordonnées dépasse 0.5, sinon 0.
Et, comme dans la vraie vie, les données d'entraînement sont imparfaites :
10 % des étiquettes sont fausses (erreurs de mesure, de saisie...).

IDÉE CENTRALE
  - jeu d'ENTRAÎNEMENT : 20 points (avec quelques étiquettes fausses),
    c'est avec eux qu'on corrige les poids ;
  - jeu de VALIDATION : 2000 points justes, JAMAIS utilisés pour corriger
    les poids ; ils servent seulement à mesurer ce que le réseau a vraiment appris.
Si la perte d'entraînement baisse mais que celle de validation remonte, le
réseau apprend par cœur (« surapprentissage »). On s'arrête alors au meilleur
moment : c'est l'ARRÊT ANTICIPÉ.
"""
import copy

import numpy as np
import torch
from torch import nn


def donnees(n, graine, bruit=0.0):
    """n points du carré [0, 1]², étiquette XOR ; une proportion `bruit` est fausse."""
    g = torch.Generator().manual_seed(graine)
    X = torch.rand(n, 2, generator=g)
    y = ((X[:, 0] > 0.5) != (X[:, 1] > 0.5)).float()[:, None]
    faux = (torch.rand(n, 1, generator=g) < bruit).float()
    return X, (y + faux) % 2, int(faux.sum())


def nouveau_reseau(graine):
    torch.manual_seed(graine)
    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())


def perte(net, X, y):
    return ((net(X) - y) ** 2).mean()


def precision(net, X, y):
    return ((net(X) > 0.5).float() == y).float().mean().item()


def entrainer(graine, epoques=1500):
    """Entraîne et renvoie les deux courbes de perte ainsi que les poids à chaque époque."""
    X, y, _ = donnees(20, graine, bruit=0.1)
    Xv, yv, _ = donnees(2000, 1000 + graine)  # points différents, étiquettes justes
    net = nouveau_reseau(graine)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    courbe_entr, courbe_val, poids = [], [], []
    for _ in range(epoques):
        opt.zero_grad()
        p = perte(net, X, y)
        p.backward()
        opt.step()
        with torch.no_grad():
            courbe_entr.append(p.item())
            courbe_val.append(perte(net, Xv, yv).item())
        poids.append(copy.deepcopy(net.state_dict()))
    return net, (X, y, Xv, yv), courbe_entr, courbe_val, poids


# ---------------------------------------------------------------
# ÉTAPE 1 : les données
# ---------------------------------------------------------------
X, y, n_faux = donnees(20, 0, bruit=0.1)
print("ÉTAPE 1 - entraînement :", len(X), "points, dont", n_faux, "avec une étiquette fausse (bruit)")
print("         validation    : 2000 points justes, jamais vus pendant l'entraînement")


# ---------------------------------------------------------------
# ÉTAPE 2 : deux courbes de perte
# ---------------------------------------------------------------
net, (X, y, Xv, yv), c_entr, c_val, poids = entrainer(graine=0)
print("\nÉTAPE 2 - perte (graine 0)")
print("         époque  entraînement  validation")
for e in [1, 25, 50, 100, 130, 200, 400, 1500]:
    print(f"         {e:6d}  {c_entr[e - 1]:12.4f}  {c_val[e - 1]:10.4f}")
# La perte d'entraînement continue de baisser, jusqu'à ~0 : le réseau finit
# par mémoriser les 20 points, y compris les étiquettes FAUSSES.
# La perte de validation, elle, baisse puis REMONTE : c'est le surapprentissage.


# ---------------------------------------------------------------
# ÉTAPE 3 : arrêt anticipé
# ---------------------------------------------------------------
# On garde les poids de l'époque où la perte de validation est la plus basse.
meilleure = int(np.argmin(c_val))
fin = copy.deepcopy(net)
net.load_state_dict(poids[meilleure])
print(f"\nÉTAPE 3 - meilleure époque = {meilleure + 1} (perte de validation {c_val[meilleure]:.4f})")
print(f"         réponses justes sur les 2000 points de validation :"
      f" arrêt à l'époque {meilleure + 1} -> {precision(net, Xv, yv):.1%},"
      f" après 1500 époques -> {precision(fin, Xv, yv):.1%}")
# Le réseau « trop » entraîné est moins bon sur des points neufs.


# ---------------------------------------------------------------
# ÉTAPE 4 : est-ce vrai pour toutes les graines ?
# ---------------------------------------------------------------
print("\nÉTAPE 4 - 10 graines : (graine : meilleure époque | précision à cette époque -> à 1500)")
meilleures, gains = [], []
for graine in range(10):
    n, (_, _, Xv, yv), _, cv, ps = entrainer(graine)
    m = int(np.argmin(cv))
    fin = copy.deepcopy(n)
    n.load_state_dict(ps[m])
    p_arret, p_fin = precision(n, Xv, yv), precision(fin, Xv, yv)
    meilleures.append(m + 1)
    gains.append(p_arret - p_fin)
    print(f"         graine {graine} : époque {m + 1:4d} | {p_arret:.1%} -> {p_fin:.1%}")
print(f"         meilleure époque médiane : {int(np.median(meilleures))}"
      f" | gain moyen de précision grâce à l'arrêt : {np.mean(gains):+.1%}")
