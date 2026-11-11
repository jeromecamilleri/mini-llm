"""
LEÇON 1 (transformer) : l'attention la plus simple
==================================================

    .venv/bin/python src/03_transformer/scripts/01_attention_simple.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
Jusqu'ici, un embedding (02_tokenizer, leçon 4) donne à un mot le MÊME vecteur
partout : « dort » est identique dans « le chat dort » et « le moteur dort ».
Or le sens d'un mot dépend de ses voisins. L'ATTENTION fait ce travail :
chaque token regarde les autres tokens de la phrase et mélange leurs
vecteurs avec le sien, plus ou moins fort selon leur intérêt.

IDÉE CENTRALE, en 3 gestes
  1. SCORES : mesurer à quel point deux tokens se ressemblent (produit scalaire).
  2. POIDS  : les transformer en pourcentages qui font 100 % (softmax).
  3. MÉLANGE : nouveau vecteur = moyenne des vecteurs, pondérée par ces poids.

VOCABULAIRE
  - PRODUIT SCALAIRE de a et b = a1*b1 + a2*b2 + ... Grand si les vecteurs
    pointent dans la même direction (rappel : le neurone, 01_fundamentals).
  - SOFTMAX : met des scores en pourcentages. Plus un score est grand, plus son
    pourcentage l'est (formule : e^score / somme des e^score).

Ici : vecteurs écrits à la main en dimension 2, pour tout voir. Dans un vrai
modèle ils sont APPRIS (leçon 4 du tokenizer). Axe 1 ~ « côté animal »,
axe 2 ~ « côté action ».
Cette version n'est pas encore l'attention d'un GPT (leçons 2 et 3 : masque et
Query/Key/Value).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import attention_simple, softmax

mots = ["le", "chat", "dort", "."]
E = np.array([
    [0.5, 0.0],  # le   : mot outil, petit vecteur
    [2.0, 0.5],  # chat : surtout « animal »
    [0.5, 2.0],  # dort : surtout « action »
    [0.0, 0.5],  # .    : ponctuation
])
np.set_printoptions(precision=2, suppress=True)

# ---------------------------------------------------------------
# ÉTAPE 1 : les scores (produit scalaire de chaque paire)
# ---------------------------------------------------------------
scores, poids, sortie = attention_simple(E)
print("ÉTAPE 1 - scores = E @ E.T (ligne = qui regarde, colonne = qui est regardé)")
print("            " + "".join(f"{m:>7s}" for m in mots))
for m, ligne in zip(mots, scores):
    print(f"         {m:>5s}" + "".join(f"{s:7.2f}" for s in ligne))
print("         exemple : dort . chat = 0.5*2.0 + 2.0*0.5 =", E[2] @ E[1])
# Ligne « dort » : le 0.25, chat 2.00, dort 4.25, « . » 1.00.
# Un token se ressemble beaucoup à lui-même (score le plus grand ici).

# ---------------------------------------------------------------
# ÉTAPE 2 : les poids (softmax de chaque ligne)
# ---------------------------------------------------------------
print("\nÉTAPE 2 - poids = softmax(scores), chaque ligne fait 100 %")
print("            " + "".join(f"{m:>7s}" for m in mots))
for m, ligne in zip(mots, poids):
    print(f"         {m:>5s}" + "".join(f"{p:7.0%}" for p in ligne))
print("         somme de chaque ligne :", poids.sum(axis=1))
# Lecture : la ligne « dort » dit « pour me comprendre, je regarde à 86 %
# moi-même, 9 % chat, 3 % « . » et 2 % le ».

# ---------------------------------------------------------------
# ÉTAPE 3 : le mélange
# ---------------------------------------------------------------
print("\nÉTAPE 3 - nouveau vecteur de « dort » = poids @ E")
ligne = poids[2]
print("         poids de dort :", {m: f"{p:.0%}" for m, p in zip(mots, ligne)})
print("         ancien vecteur :", E[2])
print("         nouveau vecteur :", sortie[2])
# Le nouveau « dort » a glissé un peu vers « chat » : [0.5, 2.0] -> [0.62, 1.78],
# il porte maintenant un peu de « animal ». Un « dort » après « chat » n'a plus
# le même vecteur qu'un « dort » après « moteur ». C'est le contexte qui entre
# dans le vecteur. Le déplacement est faible car « dort » se regarde surtout
# lui-même (86 %) : les leçons 2 et 3 corrigent cela.

# ---------------------------------------------------------------
# ÉTAPE 4 : toute la phrase d'un coup, sans boucle
# ---------------------------------------------------------------
print("\nÉTAPE 4 - tous les tokens en une seule opération")
print("         E       :", E.shape, "(4 tokens, dimension 2)")
print("         scores  :", scores.shape, "(4 x 4 paires)")
print("         sortie  :", sortie.shape, "(même forme que E : un vecteur enrichi par token)")
# Entrée et sortie ont la MÊME forme : on pourra empiler plusieurs blocs.

# ---------------------------------------------------------------
# ÉTAPE 5 : ce que fait le softmax avec des scores très différents
# ---------------------------------------------------------------
print("\nÉTAPE 5 - softmax sur des scores de plus en plus écartés")
for s in ([1, 1, 1], [1, 2, 3], [1, 2, 10]):
    print(f"         scores {s} -> poids", softmax(s).round(3))
# Scores égaux : poids égaux. Un score nettement plus grand rafle presque tout.
