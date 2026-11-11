"""
LEÇON 5 : la rétropropagation
=============================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/05_retropropagation.py

PLACE DANS LE PARCOURS
Même exemple que les leçons 3 et 4 : le XOR, réseau [2, 4, 1] (tanh, sigmoid).
Leçon 3 : on a calculé la sortie. Leçon 4 : on a mesuré l'erreur (MSE).
Ici : pour CHAQUE poids, on calcule dans quel sens le modifier pour réduire
l'erreur. Ce nombre s'appelle le GRADIENT : dPerte/dPoids.

IDÉE CENTRALE
On remonte le réseau de la sortie vers l'entrée, avec la règle de dérivation
en chaîne : l'effet d'un poids sur la perte = effet du poids sur Z, puis de Z
sur A (la dérivée de l'activation, leçon 2), puis de A sur la perte.
    dZ = dA * activation'(Z)
    dW = A_prev.T @ dZ        db = somme de dZ
    dA_prev = dZ @ W.T        (on transmet l'erreur à la couche d'avant)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès à forward.py
from activations import ACTIVATIONS
from forward import DenseNetwork, mse

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR
net = DenseNetwork([2, 4, 1], ["tanh", "sigmoid"])
N = len(X)  # nombre d'exemples

_, grad_tanh = ACTIVATIONS["tanh"]
_, grad_sigmoid = ACTIVATIONS["sigmoid"]


# ---------------------------------------------------------------
# ÉTAPE 1 : propagation avant, en gardant Z et A de chaque couche
# ---------------------------------------------------------------
# Il faudra Z1 et Z2 (pour les dérivées) et A1 (pour dW2) au retour.
def avant(net, X):
    z1 = X @ net.W[0] + net.b[0]
    a1 = np.tanh(z1)
    z2 = a1 @ net.W[1] + net.b[1]
    a2 = 1.0 / (1.0 + np.exp(-z2))
    return z1, a1, z2, a2


z1, a1, z2, a2 = avant(net, X)
print("ÉTAPE 1 - sorties :", a2.ravel().round(4), "| MSE =", round(mse(a2, y), 4))


# ---------------------------------------------------------------
# ÉTAPE 2 : propagation arrière, couche par couche
# ---------------------------------------------------------------
# Perte = moyenne((a2 - y)^2) sur les N exemples
#   -> dPerte/da2 = 2 * (a2 - y) / N
dA2 = 2 * (a2 - y) / N
# Couche 2 (sigmoid) : on traverse l'activation, puis on calcule les gradients
dZ2 = dA2 * grad_sigmoid(z2)
dW2 = a1.T @ dZ2          # forme (4, 1) comme W2
db2 = dZ2.sum(axis=0)     # forme (1,)   comme b2
# On transmet l'erreur à la couche cachée
dA1 = dZ2 @ net.W[1].T    # forme (4, 4)
# Couche 1 (tanh)
dZ1 = dA1 * grad_tanh(z1)
dW1 = X.T @ dZ1           # forme (2, 4) comme W1
db1 = dZ1.sum(axis=0)     # forme (4,)   comme b1
print("\nÉTAPE 2 - formes des gradients :",
      "dW1", dW1.shape, "| db1", db1.shape, "| dW2", dW2.shape, "| db2", db2.shape)
# Chaque gradient a la MÊME forme que le paramètre qu'il corrige.
print("         dW2 =", dW2.ravel().round(4))
# Un gradient positif : augmenter ce poids AUGMENTE la perte -> il faut le baisser.
# Un gradient négatif : augmenter ce poids BAISSE la perte -> il faut le monter.


# ---------------------------------------------------------------
# ÉTAPE 3 : vérifier le gradient avec une mesure directe
# ---------------------------------------------------------------
# Définition du gradient : si on bouge un poids de epsilon, de combien
# la perte change-t-elle ?  (perte(w + eps) - perte(w - eps)) / (2 * eps)
epsilon = 1e-5
w_test = net.W[0]      # on teste le poids W1[0, 0]
w_test[0, 0] += epsilon
perte_plus = mse(avant(net, X)[3], y)
w_test[0, 0] -= 2 * epsilon
perte_moins = mse(avant(net, X)[3], y)
w_test[0, 0] += epsilon  # on remet le poids d'origine
numerique = (perte_plus - perte_moins) / (2 * epsilon)
print("\nÉTAPE 3 - dPerte/dW1[0,0] : rétropropagation =", round(float(dW1[0, 0]), 8),
      "| mesure directe =", round(float(numerique), 8))
# Les deux valeurs sont (presque) identiques : la rétropropagation est juste.
# Elle donne TOUS les gradients en un seul passage arrière, alors que la
# mesure directe demande de recalculer le réseau deux fois par poids.


# ---------------------------------------------------------------
# ÉTAPE 4 : un pas de correction
# ---------------------------------------------------------------
# On déplace chaque poids un peu dans le sens qui BAISSE la perte :
#   poids = poids - taux_apprentissage * gradient
taux = 0.5
perte_avant = mse(a2, y)
net.W[0] -= taux * dW1
net.b[0] -= taux * db1
net.W[1] -= taux * dW2
net.b[1] -= taux * db2
perte_apres = mse(avant(net, X)[3], y)
print(f"\nÉTAPE 4 - un pas : perte {perte_avant:.4f} -> {perte_apres:.4f}")
# La perte baisse. En répétant ce pas des centaines de fois (descente de
# gradient), le réseau apprend le XOR : c'est la leçon suivante.
