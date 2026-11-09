"""
EXERCICES 1 : le neurone
========================
Remplace chaque `...` par ta réponse, puis lance :
    .venv/bin/python src/fundamentals/exercices/ex01_neurone.py

Les `assert` vérifient ta réponse : si ça plante, relis la leçon.
Ne regarde pas la correction avant d'avoir essayé !
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- Exercice 1 : calcul à la main (papier/crayon, sans code) ---------
# x = [1, 2], w = [3, -1], b = 0.5
# Quelle est la somme pondérée z ?   (écris le nombre)
z_ex1 = ...
assert z_ex1 == 1.5, "Ex1 : refais 3*1 + (-1)*2 + 0.5"

# --- Exercice 2 : devine sans calculer --------------------------------
# sigmoid(z) quand z = 0 vaut-il : 0, 0.5 ou 1 ?
sortie_ex2 = ...
assert sortie_ex2 == 0.5, "Ex2 : regarde la sortie de l'étape 2 de la leçon"

# --- Exercice 3 : écris la fonction -----------------------------------
# Complète la fonction : somme pondérée de x par w, plus b (sans activation).
def somme_ponderee(x, w, b):
    return ...


assert somme_ponderee(np.array([1.0, 2.0]), np.array([3.0, -1.0]), 0.5) == 1.5, "Ex3"

# --- Exercice 4 : le neurone complet ----------------------------------
# Utilise somme_ponderee et sigmoid pour écrire le neurone.
def neurone(x, w, b):
    return ...


sortie = neurone(np.array([1.0, 2.0]), np.array([3.0, -1.0]), 0.5)
assert abs(sortie - 0.8176) < 1e-3, f"Ex4 : attendu ~0.8176, obtenu {sortie}"

# --- Exercice 5 : question de réflexion (réponds en commentaire) ------
# Si tous les poids valent 0, de quoi dépend la sortie du neurone ?
# Réponse : ...

print("Bravo, tous les exercices passent !")
