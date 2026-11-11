"""
LEÇON 8 : l'optimiseur Adam
===========================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/08_adam.py

PLACE DANS LE PARCOURS
Même XOR, même réseau [2, 4, 1], même perte. On garde l'activation gagnante de la
leçon 7 (tanh) et on change UNE SEULE chose : la façon de corriger les poids.

QUESTION
Existe-t-il une façon de corriger les poids qui entraîne plus vite que la descente
de gradient simple de la leçon 6 ?

DEUX RÈGLES DE CORRECTION
  Descente de gradient (leçon 6) : poids = poids - taux * gradient
      même pas pour tous les poids, proportionnel au gradient du moment.
  Adam : poids = poids - taux * m / racine(v)
      m = moyenne récente du gradient (garde la direction, comme de l'élan)
      v = moyenne récente du gradient au carré (l'échelle de ce poids)
      -> chaque poids a son propre pas, de l'ordre de "taux" quelle que soit
         la taille de son gradient.

Adam est dans tools/optimiseurs.py. C'est l'optimiseur standard pour les GPT.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès aux modules
from backpropagation import descente, retropropagation
from forward import DenseNetwork
from optimiseurs import Adam

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

SEUIL_PERTE = 0.01
MAX_EPOQUES = 20000
GRAINES = range(10)


def entrainer(activation, graine, optimiseur, taux):
    """Retourne le nombre d'époques pour passer sous le seuil, ou None si échec."""
    net = DenseNetwork([2, 4, 1], [activation, "sigmoid"], seed=graine)
    adam = Adam(net, taux) if optimiseur == "adam" else None
    for epoque in range(MAX_EPOQUES + 1):
        perte, gradients = retropropagation(net, X, y)
        if perte < SEUIL_PERTE:
            return epoque
        if adam:
            adam.pas(gradients)
        else:
            descente(net, gradients, taux)
    return None


def resume(epoques):
    reussies = sorted(e for e in epoques if e is not None)
    return f"{len(reussies):>2}/{len(epoques)} {int(np.median(reussies)):>8} {reussies[-1]:>10}"


# ---------------------------------------------------------------
# ÉTAPE 1 : descente simple contre Adam, sur le réseau tanh
# ---------------------------------------------------------------
print("ÉTAPE 1 - réseau tanh, 10 entraînements par réglage")
print(f"  {'optimiseur':<22} {'réussites':>9} {'médiane':>8} {'pire cas':>10}")
for optimiseur, taux in [("descente", 0.5), ("adam", 0.01), ("adam", 0.05), ("adam", 0.1)]:
    epoques = [entrainer("tanh", g, optimiseur, taux) for g in GRAINES]
    print(f"  {optimiseur + ', taux ' + str(taux):<22} {resume(epoques)}")
# Adam avec un taux de 0.05 : ~5 fois moins d'époques que la descente simple (58 contre 318).
# Le taux compte encore : 0.01 est trop prudent ; 0.1 est le plus rapide en
# général, mais un essai sur dix est lent (le pas est trop grand pour se poser).
# Adam ne règle donc pas tout : il faut toujours choisir un taux raisonnable.


# ---------------------------------------------------------------
# ÉTAPE 2 : Adam aide-t-il aussi une activation lente ou fragile ?
# ---------------------------------------------------------------
print("\nÉTAPE 2 - même comparaison pour d'autres activations (taux Adam 0.05)")
print(f"  {'activation + optimiseur':<24} {'réussites':>9} {'médiane':>8} {'pire cas':>10}")
for activation in ["sigmoid", "relu"]:
    for optimiseur, taux in [("descente", 0.5), ("adam", 0.05)]:
        epoques = [entrainer(activation, g, optimiseur, taux) for g in GRAINES]
        print(f"  {activation + ' + ' + optimiseur:<24} {resume(epoques)}")
# sigmoid : la plus lente avec la descente (1838 époques) devient ~18 fois plus
#           rapide avec Adam (99 époques).
# relu    : plus rapide quand elle marche (63 contre 277), mais ELLE ÉCHOUE PLUS :
#           5 échecs sur 10 au lieu de 3. Les pas d'Adam sont grands dès le début,
#           ce qui tue plus de neurones, et Adam ne fait pas revivre un neurone
#           mort (son gradient est nul).
# Les neurones morts se règlent par le choix de l'activation, pas de l'optimiseur.


# ---------------------------------------------------------------
# ÉTAPE 3 : verdict pour le XOR
# ---------------------------------------------------------------
print("\nÉTAPE 3 - verdict : tanh + Adam (taux 0.05) : ~60 époques au lieu de ~320")
