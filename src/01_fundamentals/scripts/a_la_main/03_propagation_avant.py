"""
LEÇON 3 : la propagation avant d'un réseau
==========================================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/03_propagation_avant.py

PLACE DANS LE PARCOURS
On passe d'UN neurone (leçon 1) à PLUSIEURS couches de neurones. Chaque
neurone fait toujours la même chose : somme pondérée, puis activation
(leçon 2) ; on le calcule simplement pour toute une couche en une fois.

IDÉE CENTRALE
Un réseau enchaîne des couches de neurones. Chaque couche calcule :
    Z = A_prev @ W + b      (somme pondérée de toute la couche d'un coup)
    A = activation(Z)
X a la forme (n_exemples, n_entrées) : on traite tous les exemples ensemble.

La classe DenseNetwork vit dans forward.py (module réutilisé par la suite).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès à forward.py
from forward import DenseNetwork, mse

# ---------------------------------------------------------------
# ÉTAPE 1 : construire le réseau et lire les formes
# ---------------------------------------------------------------
# [2, 4, 1] : 2 entrées, 1 couche cachée de 4 neurones, 1 sortie.
net = DenseNetwork([2, 4, 1], ["tanh", "sigmoid"])
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

print("ÉTAPE 1 - formes couche par couche :")
y_pred = net.forward(X, verbose=True)
# Couche 1 : X (4, 2) @ W1 (2, 4) -> Z (4, 4) : 4 exemples, 4 neurones cachés
# Couche 2 : A1 (4, 4) @ W2 (4, 1) -> Z (4, 1) : 4 exemples, 1 sortie


# ---------------------------------------------------------------
# ÉTAPE 2 : la sortie du réseau non entraîné
# ---------------------------------------------------------------
print("\nÉTAPE 2 - sorties :", y_pred.ravel().round(4), "| cibles :", y.ravel())
print("         perte MSE :", round(mse(y_pred, y), 4))
# Toutes les sorties sont proches de 0.5 : le réseau n'a rien appris.
# Une sortie constante 0.5 donnerait une MSE de 0.25 : on est au niveau du hasard.


# ---------------------------------------------------------------
# ÉTAPE 3 : vérifier une couche à la main
# ---------------------------------------------------------------
z1 = X[0] @ net.W[0] + net.b[0]
print("\nÉTAPE 3 - Z1 manuel :", z1.round(4), "| A1 :", np.tanh(z1).round(4))
# X[0] = [0, 0] et les biais valent 0 au départ : Z1 = 0, tanh(0) = 0.
# Ensuite la couche 2 reçoit 0 et sort sigmoid(0) = 0.5 : c'est la 1re sortie.


# ---------------------------------------------------------------
# ÉTAPE 4 : et ensuite ?
# ---------------------------------------------------------------
# Les poids sont aléatoires (écart-type sqrt(2 / n_entrées), "initialisation
# de He") pour garder des activations d'amplitude raisonnable.
# Le XOR n'est pas séparable par une droite : un seul neurone ne peut pas le
# résoudre, il faut une couche cachée ET l'entraînement par rétropropagation
# (labs suivants). Ici, on ne fait que la propagation avant.
