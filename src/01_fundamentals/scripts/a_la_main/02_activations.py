"""
LEÇON 2 : les fonctions d'activation
====================================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/02_activations.py

PLACE DANS LE PARCOURS
Zoom sur l'étape 2 de la leçon 1 (l'activation). La sigmoïde y était vue
seule ; ici on la compare à trois autres : relu, leaky_relu et tanh
(définies dans activations.py, rangées dans le dictionnaire ACTIVATIONS),
et on regarde leurs dérivées.

IDÉE CENTRALE
Sans activation, empiler des couches revient à une seule opération linéaire.
L'activation ajoute la "courbure" (non-linéarité). On a aussi besoin de sa
DÉRIVÉE : la rétropropagation s'en sert pour savoir comment ajuster les poids.

Les fonctions vivent dans activations.py (module réutilisé par la suite).
Pour tracer les courbes : .venv/bin/python src/01_fundamentals/tools/activations.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès à activations.py
from activations import ACTIVATIONS

# ACTIVATIONS n'est pas une bibliothèque : c'est un simple dictionnaire défini
# dans activations.py, qui associe un nom à un couple (fonction, dérivée) :
#   ACTIVATIONS["relu"] = (relu, relu_grad)
# Les quatre activations, en formules :
#   sigmoid(z)    = 1 / (1 + e^(-z))          sortie dans ]0, 1[
#   tanh(z)       = (e^z - e^-z) / (e^z + e^-z)  sortie dans ]-1, 1[, centrée sur 0
#   relu(z)       = max(0, z)                 0 si z < 0, sinon z
#   leaky_relu(z) = z si z > 0, sinon 0.01 * z   comme relu avec une petite pente

# ---------------------------------------------------------------
# ÉTAPE 1 : valeur et dérivée en trois points
# ---------------------------------------------------------------
z = np.array([-2.0, 0.0, 2.0])
print("ÉTAPE 1 - f(z) et f'(z) pour z =", z)
for nom, (f, grad) in ACTIVATIONS.items():
    print(f"  {nom:<10} f = {f(z).round(4)}  f' = {grad(z).round(4)}")
# sigmoid   : f(0) = 0.5, f'(0) = 0.25 (sa pente maximale)
# tanh      : f(0) = 0, f'(0) = 1
# relu      : 0 pour z < 0, z sinon ; f' = 0 ou 1
# leaky_relu: comme relu, mais garde une petite pente (0.01) pour z < 0


# ---------------------------------------------------------------
# ÉTAPE 2 : la dérivée de la sigmoïde
# ---------------------------------------------------------------
# sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
s = ACTIVATIONS["sigmoid"][0](0.0)
print("\nÉTAPE 2 - sigmoid'(0) = 0.5 * 0.5 =", s * (1 - s))


# ---------------------------------------------------------------
# ÉTAPE 3 : le gradient qui disparaît
# ---------------------------------------------------------------
print("\nÉTAPE 3 - pente de la sigmoïde loin de 0 :")
sigmoid_grad = ACTIVATIONS["sigmoid"][1]
for valeur in [0, 2, 5, 10]:
    print(f"  sigmoid'({valeur:>2}) = {sigmoid_grad(float(valeur)):.6f}")
# La pente s'écrase vers 0 quand |z| est grand : le neurone n'apprend presque
# plus (gradient "qui disparaît"). tanh a le même défaut.
# relu garde une pente de 1 pour z > 0, mais 0 pour z < 0 : le neurone peut
# "mourir". leaky_relu évite ce cas avec sa petite pente.
# Les courbes correspondantes sont dans fiches/a_la_main/02_activations.md.
