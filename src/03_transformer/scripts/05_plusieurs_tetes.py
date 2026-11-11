"""
LEÇON 5 (transformer) : plusieurs têtes d'attention
===================================================

    .venv/bin/python src/03_transformer/scripts/05_plusieurs_tetes.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
Leçon 3 : une attention pose UNE question par token (Q) et reçoit UN mélange.
Mais dans « le chat dort », « dort » veut savoir QUI dort (-> chat), alors que
« chat » veut savoir de quel déterminant il est accompagné (-> le). Deux questions
différentes, une seule attention : elle doit choisir.

IDÉE CENTRALE
Une TÊTE = une attention complète (ses propres Wq, Wk, Wv, leçon 3).
On en lance PLUSIEURS EN PARALLÈLE, chacune avec ses propres matrices :
  1. chaque tête pose sa propre question et calcule ses propres poids ;
  2. on met leurs résultats bout à bout (CONCATÉNATION) ;
  3. une dernière matrice Wo les mélange.
Les têtes ne se parlent pas : chacune se spécialise pendant l'entraînement.

Ici dimension 4 : [animal, action, article, ponctuation], 2 têtes de dimension 2.
Matrices écrites à la main pour VOIR la spécialisation ; dans un vrai GPT elles
sont apprises et les rôles ne sont pas aussi nets.
  - Tête A « QUI FAIT L'ACTION ? » : un verbe cherche un animal.
  - Tête B « QUEL ARTICLE ? »      : un nom cherche son article.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import attention_multi_tetes, attention_qkv

mots = ["le", "chat", "dort", "."]
#                animal action article ponct
E = np.array([
    [0, 0, 1, 0],  # le
    [1, 0, 0, 0],  # chat
    [0, 1, 0, 0],  # dort
    [0, 0, 0, 1],  # .
], dtype=float)
np.set_printoptions(precision=2, suppress=True)

# Tête A : « qui fait l'action ? »
# Q ne s'allume que pour le trait « action » ; K ne s'allume que pour « animal » ;
# V transmet les traits animal et action.
WqA = np.array([[0, 0], [3, 0], [0, 0], [0, 0]], dtype=float)
WkA = np.array([[1, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
WvA = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)
# Tête B : « quel article ? »
# Q s'allume pour « animal » ; K pour « article » ; V transmet article et ponctuation.
WqB = np.array([[3, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
WkB = np.array([[0, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
WvB = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
Wo = np.eye(4)  # ici « ne rien mélanger de plus » ; dans un GPT, Wo est apprise


def afficher(titre, M, fmt):
    print(titre)
    print("            " + "".join(f"{m:>7s}" for m in mots))
    for m, ligne in zip(mots, M):
        print(f"         {m:>5s}" + "".join(f"{fmt(v):>7s}" for v in ligne))


# ---------------------------------------------------------------
# ÉTAPE 1 : chaque tête a ses propres poids
# ---------------------------------------------------------------
poids, sortie = attention_multi_tetes(E, [(WqA, WkA, WvA), (WqB, WkB, WvB)], Wo)
pct = lambda v: f"{v:.0%}"
afficher("ÉTAPE 1 - tête A (qui fait l'action ?) : poids", poids[0], pct)
print()
afficher("         tête B (quel article ?) : poids", poids[1], pct)
# Ligne « dort » : tête A met 81 % sur « chat » ; tête B n'a pas de question pour un
# verbe, donc répartit à égalité (33 % chacun).
# Ligne « chat » : tête B met 89 % sur « le » ; tête A n'a pas de question pour un nom
# (50/50).
# Une même phrase, deux façons de la lire.

# ---------------------------------------------------------------
# ÉTAPE 2 : concaténer, puis Wo
# ---------------------------------------------------------------
*_, sortie_a = attention_qkv(E, WqA, WkA, WvA)
*_, sortie_b = attention_qkv(E, WqB, WkB, WvB)
print("\nÉTAPE 2 - sortie de « dort » :")
print("         tête A (animal, action)   :", sortie_a[2])
print("         tête B (article, ponct.)  :", sortie_b[2])
print("         bout à bout, puis @ Wo    :", sortie[2], " = [animal, action, article, ponct.]")
# Les têtes s'ajoutent bout à bout : 2 têtes de 2 nombres = 4 nombres, la taille
# d'origine. Entrée et sortie gardent la MÊME forme (T, 4).

# ---------------------------------------------------------------
# ÉTAPE 3 : ce que chaque mot a appris sur ses voisins
# ---------------------------------------------------------------
print("\nÉTAPE 3 - vecteurs après attention [animal, action, article, ponct.] :")
for m, avant, apres in zip(mots, E, sortie):
    print(f"         {m:>5s} : {avant} -> {apres}")
# « dort » porte maintenant « animal » (81 %) : il sait qui dort.
# « chat » porte maintenant « article » (89 %) : il sait qu'il est précédé de « le ».

# ---------------------------------------------------------------
# ÉTAPE 4 : même coût qu'une seule grosse tête
# ---------------------------------------------------------------
d, n_tetes = 4, 2
d_tete = d // n_tetes
une_grosse = 3 * d * d
plusieurs = n_tetes * 3 * d * d_tete
print(f"\nÉTAPE 4 - paramètres Wq, Wk, Wv : une tête de dimension {d} = {une_grosse} ; "
      f"{n_tetes} têtes de dimension {d_tete} = {plusieurs}")
print("         GPT-2 : 12 têtes x 64 dimensions = 768. On découpe la dimension, on ne l'augmente pas.")
# On obtient donc plusieurs questions SANS payer plus de paramètres : chaque tête
# travaille dans un espace plus petit.
