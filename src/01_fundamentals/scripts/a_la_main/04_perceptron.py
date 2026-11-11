"""
LEÇON 4 : un seul neurone face au XOR
=====================================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/04_perceptron.py

PLACE DANS LE PARCOURS
Même exemple que la leçon 3 : le XOR sur (x1, x2) valant 0 ou 1.
Mais cette fois avec UN SEUL neurone (celui de la leçon 1), pas un réseau.
Question : un neurone seul peut-il apprendre le XOR ?

IDÉE CENTRALE
Pour juger un réglage (w, b), on mesure l'écart avec les cibles par une PERTE :
    MSE = moyenne((sortie - cible)^2)
Apprendre = chercher les poids qui minimisent cette perte.

Le neurone et la MSE vivent dans perceptron.py (pas d'apprentissage ici).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès à perceptron.py
from perceptron import mse, neuron

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0.0, 1.0, 1.0, 0.0])  # XOR


def sorties(w, b):
    """Sortie du neurone pour chacun des 4 exemples."""
    return np.array([neuron(x, np.array(w), b) for x in X])


# ---------------------------------------------------------------
# ÉTAPE 1 : quelques réglages (w, b) sur le XOR
# ---------------------------------------------------------------
print("ÉTAPE 1 - sorties pour [0,0] [0,1] [1,0] [1,1] (cibles : 0 1 1 0)")
for w, b in [([0.0, 0.0], 0.0), ([5.0, 5.0], -2.5), ([10.0, 10.0], -5.0), ([5.0, -5.0], 0.0)]:
    p = sorties(w, b)
    print(f"  w={w}, b={b:+.1f} -> {p.round(3)}  MSE={mse(p, y):.4f}")
# w=[0,0], b=0 : sort toujours 0.5 (indécis), MSE = 0.25 = le hasard.
# w=[5,5], b=-2.5 : un OU : sort ~1 dès qu'une entrée vaut 1. Juste sur 3 cas,
#   mais [1,1] sort 0.999 au lieu de 0.
# w=[10,10], b=-5 : le même OU, plus tranché : 3 cas parfaits, mais l'erreur sur
#   [1,1] est maximale (1) : MSE = 0.25, aucun gain sur le hasard.
# w=[5,-5], b=0 : réussit [0,1] et [1,0], mais [0,0] et [1,1] restent à 0.5.
# Chaque réglage corrige des cas et en rate d'autres.


# ---------------------------------------------------------------
# ÉTAPE 2 : chercher le meilleur réglage (recherche à l'aveugle)
# ---------------------------------------------------------------
# On essaie toutes les combinaisons (w1, w2, b) entières de -8 à 8.
valeurs = range(-8, 9)
meilleure = (np.inf, None)
for w1 in valeurs:
    for w2 in valeurs:
        for b in valeurs:
            perte = mse(sorties([w1, w2], b), y)
            if perte < meilleure[0]:
                meilleure = (perte, (w1, w2, b))
print("\nÉTAPE 2 - meilleur réglage trouvé (w1, w2, b) :", meilleure[1],
      "| MSE =", round(float(meilleure[0]), 4))
# Le meilleur réglage est w=[0,0], b=0 : MSE = 0.25, soit le hasard.
# Aucune combinaison ne fait mieux : UN neurone ne sait pas faire le XOR.
# Raison : il sépare les cas par une droite, et le XOR n'est pas séparable
# par une droite. C'est pourquoi la leçon 3 utilisait une couche cachée.
# Prochaine étape : entraîner ce réseau (rétropropagation + descente de gradient).
