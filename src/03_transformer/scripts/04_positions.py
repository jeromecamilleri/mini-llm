"""
LEÇON 4 (transformer) : les positions
=====================================

    .venv/bin/python src/03_transformer/scripts/04_positions.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
Leçon 3 : l'attention compare des tokens avec Q, K et V. Mais elle ne regarde
QUE le contenu des vecteurs, jamais la place du token dans la phrase.
Conséquence : « le chat dort » et « chat le dort » donnent à « dort » EXACTEMENT
le même vecteur, puisque les mots d'avant sont les mêmes, seulement permutés.
Pour un langage, l'ordre compte (« le chien mord l'homme » n'est pas
« l'homme mord le chien »).

IDÉE CENTRALE
Avant l'attention, on ajoute à chaque token un vecteur qui dépend de SA POSITION :
    X[i] = E[i] + P[i]
  - E : l'embedding du token (02_tokenizer, leçon 4), le même partout.
  - P : l'embedding de position, un vecteur par place 0, 1, 2, ...
  - Le même mot à deux places différentes reçoit deux vecteurs différents.
P est une table de paramètres APPRIS, comme E (c'est la solution de GPT-2).

Même exemple que les leçons précédentes. On réutilise les matrices Wq, Wk, Wv
de la leçon 3.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import ajouter_positions, attention_qkv

le, chat, dort = np.array([0.5, 0.0]), np.array([2.0, 0.5]), np.array([0.5, 2.0])
Wq = np.array([[0.0, 1.0], [1.0, 0.0]])
Wk = np.eye(2)
Wv = np.array([[1.0, 0.0], [0.0, 0.5]])
np.set_printoptions(precision=2, suppress=True)

phrase_a = ["le", "chat", "dort"]
phrase_b = ["chat", "le", "dort"]
vecteurs = {"le": le, "chat": chat, "dort": dort}
E_a = np.array([vecteurs[m] for m in phrase_a])
E_b = np.array([vecteurs[m] for m in phrase_b])


def poids_et_sortie_dernier(E):
    """Poids d'attention et nouveau vecteur du DERNIER token (« dort »)."""
    *_, poids, sortie = attention_qkv(E, Wq, Wk, Wv)
    return poids[-1], sortie[-1]


# ---------------------------------------------------------------
# ÉTAPE 1 : sans positions, l'ordre est invisible
# ---------------------------------------------------------------
print("ÉTAPE 1 - SANS positions : on regarde « dort » dans deux ordres")
for phrase, E in [(phrase_a, E_a), (phrase_b, E_b)]:
    poids, sortie = poids_et_sortie_dernier(E)
    print(f"         {' '.join(phrase):14s} poids sur {phrase} = {poids}   vecteur de dort = {sortie}")
# Les poids sont permutés (chaque mot garde SON poids), mais le vecteur final de
# « dort » est identique : le mélange est une somme, et une somme ne dépend pas
# de l'ordre des termes. Le modèle ne peut pas distinguer les deux phrases.

# ---------------------------------------------------------------
# ÉTAPE 2 : une table de positions
# ---------------------------------------------------------------
P = np.array([
    [0.0, 0.0],  # place 0
    [0.3, 0.1],  # place 1
    [0.6, 0.2],  # place 2
    [0.9, 0.3],  # place 3
])
print("\nÉTAPE 2 - table des positions P (une ligne par place), ici écrite à la main :")
for i, ligne in enumerate(P):
    print(f"         place {i} : {ligne}")
X_a, X_b = ajouter_positions(E_a, P), ajouter_positions(E_b, P)
print("         exemple : « chat » à la place 1 :", chat, "+", P[1], "=", chat + P[1])
print("                   « chat » à la place 0 :", chat, "+", P[0], "=", chat + P[0])
# Le même mot « chat » a maintenant deux vecteurs différents selon sa place.

# ---------------------------------------------------------------
# ÉTAPE 3 : avec positions, l'ordre compte
# ---------------------------------------------------------------
print("\nÉTAPE 3 - AVEC positions : mêmes deux phrases")
resultats = []
for phrase, X in [(phrase_a, X_a), (phrase_b, X_b)]:
    poids, sortie = poids_et_sortie_dernier(X)
    resultats.append(sortie)
    print(f"         {' '.join(phrase):14s} poids sur {phrase} = {poids}   vecteur de dort = {sortie}")
print("         vecteurs de « dort » identiques ?", np.allclose(*resultats))
# Les deux vecteurs de « dort » diffèrent : l'ordre est maintenant dans le calcul.

# ---------------------------------------------------------------
# ÉTAPE 4 : combien de places ? (la limite de la fenêtre de contexte)
# ---------------------------------------------------------------
T_max, d = 1024, 768  # GPT-2
print(f"\nÉTAPE 4 - GPT-2 : {T_max} places x {d} dimensions = {T_max * d:,} nombres appris".replace(",", " "))
print("         Une table a un nombre de lignes FIXE : on ne peut pas lire plus de")
print("         T_max tokens d'un coup. C'est l'origine de la « fenêtre de contexte »")
print("         (02_tokenizer, leçon 5).")
# D'autres modèles récents n'ajoutent pas un vecteur : ils font tourner Q et K
# selon la place (RoPE). Même but : rendre l'ordre visible.
