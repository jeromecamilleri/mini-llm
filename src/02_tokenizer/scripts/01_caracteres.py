"""
LEÇON 1 (tokenizer) : du texte aux nombres, caractère par caractère
===================================================================

    .venv/bin/python src/02_tokenizer/scripts/01_caracteres.py

PLACE DANS LE PARCOURS (chapitre 02_tokenizer)
Jusqu'ici notre réseau recevait des NOMBRES (x1, x2 du XOR). Un GPT reçoit du
TEXTE. Il faut donc une traduction texte <-> nombres, dans les deux sens :
  texte  --encoder-->  identifiants (entiers)  --réseau-->  identifiants  --décoder-->  texte
Cette traduction s'appelle la TOKENISATION. Un TOKEN est une unité de texte ;
le VOCABULAIRE est la liste de tous les tokens connus, chacun avec son identifiant.

Ici, le plus simple : un token = un caractère.
Mini-corpus (écrit à la main, quelques mots seulement) :
    "le chat dort. le chien dort aussi. le chat mange."
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès à vocabulary.py
from vocabulary import Vocabulaire

texte = "le chat dort. le chien dort aussi. le chat mange."


# ---------------------------------------------------------------
# ÉTAPE 1 : le vocabulaire
# ---------------------------------------------------------------
# On découpe le texte en caractères, puis on garde chaque caractère distinct
# (triés) : son identifiant est sa position dans la liste.
vocab = Vocabulaire.depuis_unites(texte)
print("ÉTAPE 1 - texte :", repr(texte), f"({len(texte)} caractères)")
print("         vocabulaire :", len(vocab), "tokens")
print("         ", {t: i for t, i in vocab.token_vers_id.items()})
# L'espace est un token comme les autres. L'ordre des identifiants est
# arbitraire (ordre alphabétique) : 'a' = 2 et 'c' = 3 sont voisins, 'a' et 'u' = 16
# sont éloignés, sans aucun rapport avec leur sens.


# ---------------------------------------------------------------
# ÉTAPE 2 : encoder, puis décoder
# ---------------------------------------------------------------
ids = vocab.encoder_unites("le chat")
print("\nÉTAPE 2 - encoder 'le chat' :", ids)
retour = "".join(vocab.decoder_ids(ids))
print("         décoder ces identifiants :", repr(retour))
# Aller-retour exact : on ne perd rien. Le réseau, lui, ne voit que la liste de nombres.
tout = vocab.encoder_unites(texte)
print("         tout le corpus :", len(tout), "identifiants, aller-retour exact :",
      "".join(vocab.decoder_ids(tout)) == texte)


# ---------------------------------------------------------------
# ÉTAPE 3 : un caractère inconnu
# ---------------------------------------------------------------
# Le vocabulaire est figé : il ne contient que ce qu'il a vu dans le corpus.
try:
    vocab.encoder_unites("le chat ronronne é")
except KeyError as erreur:
    print("\nÉTAPE 3 - encoder 'le chat ronronne é' : caractère inconnu", erreur)
# 'é' n'est pas dans notre mini-corpus. Avec le français réel, il faudra un
# vocabulaire qui contient les accents, la ponctuation, les majuscules...
# On verra deux solutions : un grand vocabulaire de caractères (étape suivante),
# ou des SOUS-MOTS (des morceaux de mots, trouvés par la méthode BPE) qui
# découpent n'importe quel mot en morceaux connus. On les construit à la leçon 3.


# ---------------------------------------------------------------
# ÉTAPE 4 : un identifiant n'a pas de « sens »
# ---------------------------------------------------------------
# Donner le NOMBRE 3 à un réseau comme entrée est trompeur : il croirait
# que 'c' (3) est « entre » 'a' (2) et 'e' (5). Une solution simple : le
# ONE-HOT, un vecteur de la taille du vocabulaire avec un seul 1.
un_chat = np.eye(len(vocab))[vocab.encoder_unites("le chat")]
print("\nÉTAPE 4 - 'le chat' en one-hot : forme", un_chat.shape,
      "= (7 caractères, 17 tokens du vocabulaire)")
print("         la lettre 'l' :", un_chat[0].astype(int))
# Chaque ligne est un vecteur de 0 avec un seul 1, à la place de l'identifiant.
# Tous les caractères sont à égale « distance » : le réseau ne sait pas encore que
# 'a' et 'e' sont deux voyelles. C'est ce que les EMBEDDINGS (leçon 4 de ce chapitre)
# apprendront : remplacer ce vecteur creux par un petit vecteur appris.


# ---------------------------------------------------------------
# ÉTAPE 5 : ce que ça coûte
# ---------------------------------------------------------------
mots = texte.replace(".", " .").split()
print("\nÉTAPE 5 - le même texte, deux découpages")
print(f"         caractères : {len(texte):3d} tokens, vocabulaire de {len(vocab)}")
print(f"         mots       : {len(mots):3d} tokens, vocabulaire de {len(set(mots))}")
# Caractères : un petit vocabulaire, mais des séquences LONGUES (le réseau doit
# relier beaucoup de tokens pour « voir » un mot).
# Mots : des séquences courtes, mais un vocabulaire qui grossit avec le corpus
# et ne sait rien faire d'un mot inconnu.
# Les SOUS-MOTS sont un compromis : des morceaux de mots (« chats » = « chat » + « s »),
# trouvés par la méthode BPE (on la construit à la leçon 3). C'est ce que font les vrais GPT.
