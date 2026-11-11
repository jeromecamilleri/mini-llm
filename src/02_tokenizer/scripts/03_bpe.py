"""
LEÇON 3 (tokenizer) : BPE, les sous-mots trouvés automatiquement
================================================================

    .venv/bin/python src/02_tokenizer/scripts/03_bpe.py

PLACE DANS LE PARCOURS (chapitre 02_tokenizer)
Leçon 1 : caractères (vocabulaire minuscule, séquences longues).
Leçon 2 : mots (séquences courtes, mais mots inconnus et vocabulaire qui explose).
Ici : les SOUS-MOTS, des morceaux de mots, trouvés par BPE (Byte Pair Encoding).

IDÉE CENTRALE : on part des caractères, puis on répète :
  1. compter chaque PAIRE de tokens voisins dans le corpus ;
  2. fusionner la paire la plus fréquente en UN nouveau token ;
  3. ajouter ce token au vocabulaire.
Nombre de fusions choisi = taille finale du vocabulaire (caractères + fusions).
Pour encoder un texte neuf, on REJOUE les fusions dans le même ordre.

Même corpus que la leçon 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from bpe import BPE, compter_paires, decouper_morceaux
from collections import Counter

corpus = (
    "le chat dort. le chien dort aussi. le chat mange. le chien mange un os. "
    "un oiseau chante. le chat regarde l'oiseau. le chien court. l'oiseau dort."
)


# ---------------------------------------------------------------
# ÉTAPE 1 : le point de départ, des caractères
# ---------------------------------------------------------------
# Le texte est coupé en morceaux : un mot garde l'espace qui le précède
# (« ␣chat »), la ponctuation est à part. Les fusions ne traversent jamais
# un morceau : on ne fusionnera pas la fin d'un mot avec le début du suivant.
# (Le tout premier « le » n'a pas d'espace devant : c'est un morceau à part de « ␣le ».)
morceaux = Counter(tuple(m) for m in decouper_morceaux(corpus))
print("ÉTAPE 1 - 4 morceaux parmi", len(morceaux), "distincts :")
for tokens, frequence in list(morceaux.items())[:4]:
    print(f"         {frequence} x {list(tokens)}")
paires = compter_paires(morceaux)
print("         paires les plus fréquentes :",
      [(a + b, n) for (a, b), n in paires.most_common(5)])
# Chaque morceau est une suite de caractères ; on compte les paires voisines,
# pondérées par le nombre de fois où le morceau apparaît.


# ---------------------------------------------------------------
# ÉTAPE 2 : les premières fusions
# ---------------------------------------------------------------
bpe = BPE(corpus, n_fusions=15)
print("\nÉTAPE 2 - les 15 fusions, dans l'ordre :")
for i, ((a, b), frequence) in enumerate(bpe.historique, 1):
    print(f"         {i:2d}. {a!r} + {b!r} -> {a + b!r}   (vue {frequence} fois)")
# Les premières paires fusionnées sont les plus fréquentes du corpus ; peu à peu
# apparaissent des syllabes, puis des mots entiers (« ␣le », « ␣chat »...).
# Chaque fusion ajoute UN token au vocabulaire.


# ---------------------------------------------------------------
# ÉTAPE 3 : le vocabulaire obtenu
# ---------------------------------------------------------------
n_car = len(bpe.vocab) - len(bpe.fusions)
print(f"\nÉTAPE 3 - vocabulaire : {n_car} caractères + {len(bpe.fusions)} fusions"
      f" = {len(bpe.vocab)} tokens")
print("         tokens ajoutés :", bpe.vocab.id_vers_token[n_car:])
# Les caractères restent dans le vocabulaire : ils servent de filet de sécurité
# pour les mots jamais vus.


# ---------------------------------------------------------------
# ÉTAPE 4 : encoder et décoder
# ---------------------------------------------------------------
for phrase in ["le chat dort.", "le chien ronronne."]:
    morceaux_phrase = bpe.decouper(phrase)
    ids = bpe.encoder(phrase)
    print(f"\nÉTAPE 4 - {phrase!r}")
    print("         sous-mots :", morceaux_phrase)
    print("         identifiants :", ids, "| décodé :", repr(bpe.decoder(ids)))
# « le chat dort. » (déjà vu) tient en très peu de tokens.
# « ronronne » n'existe pas dans le corpus, mais il n'est PAS inconnu : il est
# découpé en morceaux connus, jusqu'aux caractères au pire. Rien n'est perdu
# (leçon 2 : <inc>). Le décodage redonne exactement le texte.


# ---------------------------------------------------------------
# ÉTAPE 5 : combien de fusions ?
# ---------------------------------------------------------------
neuf = "le canard dort. le chien nage."
print("\nÉTAPE 5 - effet du nombre de fusions (tokens pour le corpus | pour un texte neuf)")
print("         fusions   vocabulaire   corpus   texte neuf")
for n in [0, 5, 15, 30, 60]:
    b = BPE(corpus, n)
    print(f"         {n:7d}   {len(b.vocab):11d}   {len(b.decouper(corpus)):6d}   {len(b.decouper(neuf)):10d}")
# 0 fusion = leçon 1 (caractères). Plus on fusionne, plus les séquences raccourcissent,
# mais plus le vocabulaire grossit. Sur le texte neuf le gain est plus faible :
# il contient des mots que le corpus n'a jamais vus.
# À toi de choisir le compromis : c'est un réglage (hyperparamètre) du projet.
