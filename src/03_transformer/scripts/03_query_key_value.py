"""
LEÇON 3 (transformer) : Query, Key, Value
=========================================

    .venv/bin/python src/03_transformer/scripts/03_query_key_value.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
Leçon 1 : les scores venaient de E @ E.T. E est le tableau des vecteurs (une ligne
par token) ; E @ E.T fait le produit scalaire de TOUTES les paires, en une
opération. Défaut : un vecteur ressemble toujours beaucoup à lui-même (« dort »
avec « dort » = 4,25, le plus grand score), donc « dort » se regardait à 86 %. Pas très utile : « dort » veut surtout savoir QUI
dort, c'est-à-dire regarder « chat ».
Ici : au lieu de comparer les vecteurs bruts, on fabrique trois versions de
chaque token grâce à trois petites matrices, qui seront APPRISES :

  - QUERY (Q, « la question »)   : ce que ce token CHERCHE chez les autres.
  - KEY   (K, « l'étiquette »)   : ce que ce token PRÉSENTE pour être trouvé.
  - VALUE (V, « le contenu »)    : ce que ce token APPORTE s'il est choisi.

IMAGE : une bibliothèque. Tu arrives avec une question (Q). Chaque livre a une
étiquette sur sa tranche (K). Tu compares ta question aux étiquettes : plus
elles correspondent, plus tu ouvres le livre. Ce que tu lis dedans, c'est son
contenu (V), qui n'est pas forcément l'étiquette.

FORMULES (E : un vecteur par token)
  Q = E @ Wq      K = E @ Wk      V = E @ Wv
  scores = Q @ K.T / racine(dk)     puis masque causal, puis softmax (leçons 1 et 2)
  sortie = poids @ V
Wq, Wk, Wv sont des paramètres comme W1 en 01_fundamentals : on les corrige par
rétropropagation. Ici on les écrit à la main pour VOIR l'effet.

Même exemple : « le chat dort . »
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import attention_causale, attention_qkv

mots = ["le", "chat", "dort", "."]
E = np.array([
    [0.5, 0.0],  # le
    [2.0, 0.5],  # chat : axe 1 = « animal »
    [0.5, 2.0],  # dort : axe 2 = « action »
    [0.0, 0.5],  # .
])
np.set_printoptions(precision=2, suppress=True)


def afficher(titre, M, fmt=lambda v: f"{v:.2f}"):
    print(titre)
    print("            " + "".join(f"{m:>7s}" for m in mots))
    for m, ligne in zip(mots, M):
        print(f"         {m:>5s}" + "".join(f"{fmt(v):>7s}" for v in ligne))


# ---------------------------------------------------------------
# ÉTAPE 1 : trois matrices, trois versions de chaque token
# ---------------------------------------------------------------
Wq = np.array([[0.0, 1.0], [1.0, 0.0]])  # échange les deux axes : « je cherche l'AUTRE axe »
Wk = np.eye(2)                           # l'étiquette = le vecteur tel quel
Wv = np.array([[1.0, 0.0], [0.0, 0.5]])  # le contenu transmis : axe 2 réduit de moitié
Q, K, V, scores, poids, sortie = attention_qkv(E, Wq, Wk, Wv)
print("ÉTAPE 1 - Q = E @ Wq, K = E @ Wk, V = E @ Wv")
print("            " + " ".join(f"{m:>5s}" for m in mots))
print("         E :", " ".join(str(v) for v in E))
print("         Q :", " ".join(str(v) for v in Q))
print("         K :", " ".join(str(v) for v in K))
print("         V :", " ".join(str(v) for v in V))
# Wq échange les axes : la question de « dort » (axe 2 fort) devient [2.0, 0.5],
# qui ressemble à l'étiquette K de « chat » ([2.0, 0.5]). Un verbe cherche un animal.

# ---------------------------------------------------------------
# ÉTAPE 2 : scores = Q @ K.T / racine(dk), avec le masque causal
# ---------------------------------------------------------------
print()
afficher("ÉTAPE 2 - scores (Q @ K.T / racine(2), masque causal)", scores,
         lambda v: "-inf" if np.isinf(v) else f"{v:.2f}")
print("         exemple : Q(dort) . K(chat) = 2.0*2.0 + 0.5*0.5 =", Q[2] @ K[1], "; divisé par racine(2) =", round(Q[2] @ K[1] / np.sqrt(2), 2))
# Pourquoi diviser par racine(dk) ? Un produit scalaire grandit avec la dimension :
# avec des vecteurs de 768 nombres, les scores seraient énormes et le softmax
# donnerait ~100 % à un seul token. On les ramène à une taille raisonnable.

# ---------------------------------------------------------------
# ÉTAPE 3 : poids, avant et après Q/K/V
# ---------------------------------------------------------------
_, poids_ancien, _ = attention_causale(E)
print()
afficher("ÉTAPE 3 - poids avec Q/K/V", poids, lambda v: f"{v:.0%}")
print("\n         ligne « dort » :")
print("           leçon 2 (E @ E.T) :", {m: f"{p:.0%}" for m, p in zip(mots[:3], poids_ancien[2][:3])})
print("           avec Q/K/V        :", {m: f"{p:.0%}" for m, p in zip(mots[:3], poids[2][:3])})
# « dort » regardait surtout lui-même (89 %). Avec Q/K/V, il regarde surtout
# « chat » (77 %) : il va chercher QUI dort.

# ---------------------------------------------------------------
# ÉTAPE 4 : le mélange porte sur V, pas sur E
# ---------------------------------------------------------------
print("\nÉTAPE 4 - sortie = poids @ V")
print("         nouveau vecteur de « dort » :", sortie[2])
print("         calcul : 8% de V(le) + 77% de V(chat) + 16% de V(dort)")
# La comparaison (Q, K) et le contenu transmis (V) sont séparés : un token peut
# être trouvé grâce à son étiquette et apporter autre chose que son étiquette.

# ---------------------------------------------------------------
# ÉTAPE 5 : les matrices sont des paramètres à apprendre
# ---------------------------------------------------------------
n = Wq.size + Wk.size + Wv.size
print(f"\nÉTAPE 5 - paramètres : Wq {Wq.shape} + Wk {Wk.shape} + Wv {Wv.shape} = {n} nombres")
print("         Ici écrits à la main. Dans le GPT : initialisés au hasard puis appris,")
print("         comme W1 et b1 de 01_fundamentals (leçons 5 et 9).")
print(f"         Avec d = 768 (GPT-2) : 3 x 768 x 768 = {3 * 768 * 768:,} nombres par tête.".replace(",", " "))
