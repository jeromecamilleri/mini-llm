"""
LEÇON 2 (transformer) : le masque causal
========================================

    .venv/bin/python src/03_transformer/scripts/02_masque_causal.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
Leçon 1 : chaque token regarde TOUS les autres, y compris ceux qui le suivent.
Problème : on entraîne le GPT à prédire le mot SUIVANT (02_tokenizer, leçon 5).
Si, pour prédire « dort », il peut déjà lire « dort » ou « . » plus loin dans la
fenêtre, il TRICHE : il recopie la réponse au lieu de la deviner.

IDÉE CENTRALE
Le MASQUE CAUSAL interdit à un token de regarder les tokens qui viennent après lui.
  - Token 0 : ne voit que lui-même.  Token 1 : les tokens 0 et 1.  Etc.
  - En pratique : on remplace les scores interdits par -infini AVANT le softmax.
    e^(-infini) = 0, donc leur poids devient exactement 0 %.

Même exemple que la leçon 1 : « le chat dort . »

POURQUOI CODER UN MASQUE ? (dans un réseau, on code la structure, pas la règle)
Un réseau APPREND : si une porte est ouverte vers la réponse, il la prend, c'est
le chemin le plus facile. Sans masque, il apprendrait « recopie le mot suivant »
(perte proche de 0), puis serait inutile à la GÉNÉRATION, où ce mot n'existe pas
encore. Le masque ferme la porte : l'entraînement ressemble à l'usage réel.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import attention_causale, attention_simple, masque_causal

mots = ["le", "chat", "dort", "."]
E = np.array([
    [0.5, 0.0],  # le
    [2.0, 0.5],  # chat
    [0.5, 2.0],  # dort
    [0.0, 0.5],  # .
])
np.set_printoptions(precision=2, suppress=True)


def afficher(titre, M, fmt):
    print(titre)
    print("            " + "".join(f"{m:>7s}" for m in mots))
    for m, ligne in zip(mots, M):
        print(f"         {m:>5s}" + "".join(f"{fmt(v):>7s}" for v in ligne))


# ---------------------------------------------------------------
# ÉTAPE 1 : le masque (qui a le droit de regarder qui)
# ---------------------------------------------------------------
afficher("ÉTAPE 1 - masque (1 = permis, . = interdit)", masque_causal(4), lambda v: "1" if v else ".")
# Une matrice triangulaire : la ligne i autorise les colonnes 0 à i.

# ---------------------------------------------------------------
# ÉTAPE 2 : scores masqués, puis poids
# ---------------------------------------------------------------
scores, poids, sortie = attention_causale(E)
print()
afficher("ÉTAPE 2 - scores masqués (-inf = interdit)", scores, lambda v: "-inf" if np.isinf(v) else f"{v:.2f}")
print()
afficher("         poids après softmax", poids, lambda v: f"{v:.0%}")
# Les poids interdits valent 0 %, et chaque ligne fait toujours 100 %.
# Le premier token (« le ») n'a qu'un seul choix : lui-même, à 100 %.

# ---------------------------------------------------------------
# ÉTAPE 3 : avant et après, pour « dort »
# ---------------------------------------------------------------
_, poids_libre, sortie_libre = attention_simple(E)
print("\nÉTAPE 3 - ligne « dort » (3e token)")
print("         sans masque :", {m: f"{p:.0%}" for m, p in zip(mots, poids_libre[2])})
print("         avec masque :", {m: f"{p:.0%}" for m, p in zip(mots, poids[2])})
# Sans masque, 3 % du poids allait à « . », qui est le FUTUR de « dort ».

# ---------------------------------------------------------------
# ÉTAPE 4 : vérifier qu'un token ne voit pas le futur
# ---------------------------------------------------------------
# Test : on change le DERNIER token (« . » devient un vecteur très différent)
# et on regarde si les sorties des tokens précédents bougent.
E2 = E.copy()
E2[3] = [5.0, -3.0]
_, _, sortie2 = attention_causale(E2)
_, _, libre2 = attention_simple(E2)
print("\nÉTAPE 4 - on remplace le dernier token et on compare les sorties des autres")
print("         avec masque : sorties de le/chat/dort changées ?", not np.allclose(sortie[:3], sortie2[:3]))
print("         sans masque : sorties de le/chat/dort changées ?", not np.allclose(sortie_libre[:3], libre2[:3]))
# Avec masque : NON, le passé ne dépend pas du futur. C'est la propriété voulue.
# Sans masque : OUI, la modification du futur se propage dans le passé = triche possible.

# ---------------------------------------------------------------
# ÉTAPE 5 : une fenêtre = T exercices en une seule passe
# ---------------------------------------------------------------
print("\nÉTAPE 5 - ce que « voit » chaque position grâce au masque")
for i, m in enumerate(mots):
    print(f"         position {i} ({m:>4s}) voit : {' '.join(mots[:i + 1])}")
# On retrouve exactement les T exercices de 02_tokenizer, leçon 5 :
# « le » -> chat, « le chat » -> dort, etc. Le masque les fabrique tous
# dans UNE SEULE opération, sans jamais montrer la réponse.
