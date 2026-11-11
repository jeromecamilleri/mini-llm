"""
LEÇON 2 (tokenizer) : le tokenizer de mots
==========================================

    .venv/bin/python src/02_tokenizer/scripts/02_mots.py

PLACE DANS LE PARCOURS (chapitre 02_tokenizer)
Leçon 1 : un token = un caractère (petit vocabulaire, longues séquences).
Ici : un token = un MOT (séquences courtes, mais vocabulaire plus grand).
On voit surtout le gros défaut des mots : que faire d'un mot qu'on n'a JAMAIS
vu dans le corpus ? Le caractère, lui, n'a pas ce problème.

On garde le même style de texte : de petites phrases sur des animaux.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from tokenizer_basic import TokenizerMots, decouper_mots
from vocabulary import Vocabulaire

corpus = (
    "le chat dort. le chien dort aussi. le chat mange. le chien mange un os. "
    "un oiseau chante. le chat regarde l'oiseau. le chien court. l'oiseau dort."
)


# ---------------------------------------------------------------
# ÉTAPE 1 : découper en mots
# ---------------------------------------------------------------
# Règles choisies : tout en minuscules, un mot = suite de lettres, chaque
# signe de ponctuation est un token à part. (Choix arbitraires : un autre
# tokenizer pourrait garder les majuscules, ou découper « l'oiseau » autrement.)
mots = decouper_mots(corpus)
print("ÉTAPE 1 - les", len(mots), "premiers mots :", mots[:12])


# ---------------------------------------------------------------
# ÉTAPE 2 : le vocabulaire, avec un jeton « inconnu »
# ---------------------------------------------------------------
tok = TokenizerMots(corpus)
print("\nÉTAPE 2 - vocabulaire :", len(tok.vocab), "tokens (dont <inc>)")
print("         ", tok.vocab.id_vers_token)
# <inc> (identifiant 0) est un token SPÉCIAL : il remplace tout mot absent du corpus.


# ---------------------------------------------------------------
# ÉTAPE 3 : encoder, décoder, et le mot inconnu
# ---------------------------------------------------------------
for phrase in ["le chien dort.", "le chat ronronne."]:
    ids = tok.encoder(phrase)
    print(f"\nÉTAPE 3 - {phrase!r} -> {ids} -> {tok.decoder(ids)!r}")
# « ronronne » n'est pas dans le corpus : il devient <inc>, et le mot est PERDU
# à jamais (le décodage ne peut plus le retrouver). Pour le caractère : aucun
# souci, 'r', 'o', 'n'... sont tous connus (leçon 1).


# ---------------------------------------------------------------
# ÉTAPE 4 : mots contre caractères sur un texte neuf
# ---------------------------------------------------------------
# On encode un texte que le tokenizer n'a jamais vu.
neuf = "le canard nage. le chat dort. un poisson nage."
ids_mots = tok.encoder(neuf)
inconnus = sum(1 for i in ids_mots if i == 0)
caracteres = Vocabulaire.depuis_unites(corpus)
perdus = [c for c in neuf if c not in caracteres.token_vers_id]
print(f"\nÉTAPE 4 - texte neuf : {neuf!r}")
print(f"         mots       : {len(ids_mots)} tokens, dont {inconnus} inconnus "
      f"({inconnus / len(ids_mots):.0%})")
print(f"         caractères : {len(neuf)} tokens, dont {len(perdus)} inconnus "
      f"({len(perdus) / len(neuf):.0%})")
# Avec si peu de texte, beaucoup de mots sont inconnus ; le caractère s'en sort
# car l'alphabet est vite complet. Avec un corpus immense, les mots inconnus
# deviennent rares mais ne disparaissent jamais (noms propres, fautes, mots rares).


# ---------------------------------------------------------------
# ÉTAPE 5 : comment le vocabulaire grandit
# ---------------------------------------------------------------
print("\nÉTAPE 5 - vocabulaire après N tokens lus (corpus de", len(mots), "mots)")
print("         mots lus   mots distincts   caractères lus   caractères distincts")
for n in [10, 20, len(mots)]:
    m = mots[:n]
    texte_n = " ".join(m)
    print(f"         {n:8d}   {len(set(m)):14d}   {len(texte_n):14d}   {len(set(texte_n)):20d}")
# Les mots distincts continuent de croître avec le texte ; les caractères
# distincts se stabilisent vite (~30 pour le français). C'est pour cela que
# les vrais GPT n'utilisent ni l'un ni l'autre, mais des SOUS-MOTS : des morceaux
# de mots (« chats » = « chat » + « s »). La méthode qui les trouve s'appelle BPE
# (Byte Pair Encoding) : on part des caractères et on fusionne la paire voisine
# la plus fréquente, encore et encore. On la construit à la leçon suivante.
