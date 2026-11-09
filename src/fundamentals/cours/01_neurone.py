"""
LEÇON 1 : le neurone artificiel
================================

Lance ce fichier, lis la sortie, puis lis le code ligne par ligne.
    .venv/bin/python src/fundamentals/cours/01_neurone.py

IDÉE CENTRALE
Un neurone est une petite fonction qui :
  1. prend des nombres en entrée (x),
  2. les multiplie chacun par un "poids" (w) = l'importance de cette entrée,
  3. additionne tout, plus un "biais" (b) = un décalage,
  4. passe le résultat dans une "activation" qui écrase la valeur.

    sortie = activation( w1*x1 + w2*x2 + ... + b )

Un réseau de neurones, ce sont des milliers de ces petites fonctions
branchées ensemble. Apprendre = trouver les bonnes valeurs de w et b.
"""
import numpy as np  # NumPy : calcul sur des tableaux de nombres

# ---------------------------------------------------------------
# ÉTAPE 1 : une somme pondérée à la main
# ---------------------------------------------------------------
x = np.array([2.0, 3.0])    # deux entrées
w = np.array([0.5, -1.0])   # un poids par entrée
b = 1.0                     # le biais

# Calcul "à la main" :  0.5*2.0 + (-1.0)*3.0 + 1.0
z = w[0] * x[0] + w[1] * x[1] + b
print("ÉTAPE 1 - somme pondérée à la main :", z)

# np.dot fait exactement la même chose (produit scalaire) en une ligne.
z2 = np.dot(w, x) + b
print("         avec np.dot               :", z2)
# -> même résultat : -1.0. z s'appelle la "pré-activation".


# ---------------------------------------------------------------
# ÉTAPE 2 : l'activation
# ---------------------------------------------------------------
# Sans activation, empiler des neurones ne ferait que des calculs
# linéaires. L'activation ajoute de la "courbure" (non-linéarité).
# La sigmoïde écrase n'importe quel nombre dans l'intervalle ]0, 1[ :
#   très négatif -> proche de 0 ; 0 -> 0.5 ; très positif -> proche de 1
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


print("\nÉTAPE 2 - la sigmoïde écrase les valeurs :")
for valeur in [-10, -1, 0, 1, 10]:
    print(f"  sigmoid({valeur:>3}) = {sigmoid(valeur):.4f}")


# ---------------------------------------------------------------
# ÉTAPE 3 : le neurone complet = étape 1 + étape 2
# ---------------------------------------------------------------
def neurone(x, w, b):
    z = np.dot(w, x) + b   # somme pondérée
    return sigmoid(z)      # activation


print("\nÉTAPE 3 - neurone complet :", neurone(x, w, b))
# z = -1.0 donc sigmoid(-1.0) = 0.2689


# ---------------------------------------------------------------
# ÉTAPE 4 : que font les poids et le biais ?
# ---------------------------------------------------------------
print("\nÉTAPE 4 - effet du biais (mêmes x et w) :")
for biais in [-5, 0, 1, 5]:
    print(f"  b = {biais:>2} -> sortie = {neurone(x, w, biais):.4f}")
# Le biais décale la sortie : plus il est grand, plus le neurone "s'active".

print("\n  effet du poids w1 (w2 = -1, b = 1) :")
for w1 in [-2, 0, 0.5, 2]:
    print(f"  w1 = {w1:>4} -> sortie = {neurone(x, np.array([w1, -1.0]), 1.0):.4f}")
# Un poids positif grand renforce son entrée, négatif l'inhibe.
