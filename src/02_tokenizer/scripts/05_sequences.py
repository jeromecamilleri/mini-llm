"""
LEÇON 5 (tokenizer) : les séquences d'entraînement
==================================================

    .venv/bin/python src/02_tokenizer/scripts/05_sequences.py

PLACE DANS LE PARCOURS (chapitre 02_tokenizer, dernière leçon)
Leçon 4 : on prédisait le mot suivant à partir d'UN seul mot.
Un GPT lit PLUSIEURS tokens avant de prédire : il lui faut des exemples du type
« voici les T derniers tokens -> voici ce qui suit ».
Ici : découper un long texte (une suite d'identifiants) en tels exemples.

VOCABULAIRE
  - FENÊTRE DE CONTEXTE (taille T) : le nombre de tokens que le modèle voit d'un coup.
  - ENTRÉE x : T tokens consécutifs. CIBLE y : les MÊMES tokens décalés d'un cran.
  - BATCH : un petit paquet d'exemples traités ensemble (leçon 10 : DataLoader).

IDÉE CENTRALE
  x = ids[i : i+T]      y = ids[i+1 : i+T+1]
À chaque position de la fenêtre, la cible est le token suivant : UNE fenêtre
de T tokens donne donc T exercices de prédiction d'un coup.

Corpus : le même que la leçon 4 (`corpus_animaux.txt`), tokenisé en mots (leçon 2).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from tokenizer_basic import TokenizerMots

corpus = (Path(__file__).resolve().parents[1] / "corpus_animaux.txt").read_text(encoding="utf-8").strip()
tok = TokenizerMots(corpus)
mots = tok.vocab.id_vers_token
ids = torch.tensor(tok.encoder(corpus))
N = len(ids)
T = 4  # taille de la fenêtre de contexte (un vrai GPT : 256 à plusieurs milliers)


def decoder(suite):
    return " ".join(mots[i] for i in suite.tolist())


# ---------------------------------------------------------------
# ÉTAPE 1 : une fenêtre = une entrée et une cible décalée
# ---------------------------------------------------------------
print("CORPUS (corpus_animaux.txt) :")
print("        ", corpus)
print(f"\nÉTAPE 1 - le corpus fait {N} tokens ; fenêtre T = {T}")
i = 0
x, y = ids[i:i + T], ids[i + 1:i + T + 1]
print("         x =", x.tolist(), "->", decoder(x))
print("         y =", y.tolist(), "->", decoder(y))
# y est x décalé d'un cran : y[k] est le token qui SUIT x[k].


# ---------------------------------------------------------------
# ÉTAPE 2 : les T exercices cachés dans une fenêtre
# ---------------------------------------------------------------
print("\nÉTAPE 2 - les exercices de prédiction de cette fenêtre :")
for k in range(T):
    print(f"         voit [{decoder(x[:k + 1])}]  ->  doit prédire '{mots[y[k].item()]}'")
# Le modèle ne doit JAMAIS voir la suite : pour prédire 'dort', il n'a que 'le chat'.
# Cette règle sera imposée par un « masque » dans l'attention (chapitre 03_transformer).


# ---------------------------------------------------------------
# ÉTAPE 3 : combien de fenêtres ? (on glisse d'un token à la fois)
# ---------------------------------------------------------------
n_fenetres = N - T
print(f"\nÉTAPE 3 - fenêtres possibles : {N} - {T} = {n_fenetres}")
print(f"         exercices de prédiction : {n_fenetres} x {T} = {n_fenetres * T}")
# Les fenêtres se chevauchent : le même texte sert plusieurs fois, sous des angles différents.


# ---------------------------------------------------------------
# ÉTAPE 4 : séparer entraînement et validation (leçon 11)
# ---------------------------------------------------------------
coupure = int(0.9 * N)
entrainement, validation = ids[:coupure], ids[coupure:]
print(f"\nÉTAPE 4 - entraînement : {len(entrainement)} tokens, validation : {len(validation)} tokens")
print("         validation =", decoder(validation))
# On coupe le TEXTE en deux blocs AVANT de fabriquer les fenêtres. Sinon une fenêtre
# d'entraînement et une de validation se chevaucheraient : le modèle aurait déjà
# « vu » la validation, et la mesure serait trop optimiste.


# ---------------------------------------------------------------
# ÉTAPE 5 : fabriquer un batch au hasard
# ---------------------------------------------------------------
def tirer_batch(donnees, taille_batch, T, generateur):
    """taille_batch fenêtres prises à des positions tirées au hasard."""
    debuts = torch.randint(0, len(donnees) - T, (taille_batch,), generator=generateur)
    x = torch.stack([donnees[d:d + T] for d in debuts])
    y = torch.stack([donnees[d + 1:d + T + 1] for d in debuts])
    return x, y


g = torch.Generator().manual_seed(0)
xb, yb = tirer_batch(entrainement, 3, T, g)
print(f"\nÉTAPE 5 - un batch de 3 fenêtres : x de forme {tuple(xb.shape)}, y de forme {tuple(yb.shape)}")
for k in range(3):
    print(f"         {decoder(xb[k]):28s} -> {decoder(yb[k])}")
# Forme (B, T) = (nombre de fenêtres, tokens par fenêtre). Le chapitre 03_transformer prendra
# exactement ce tenseur en entrée : embedding -> (B, T, dimension).


# ---------------------------------------------------------------
# ÉTAPE 6 : même chose avec les sous-mots (BPE, leçon 3)
# ---------------------------------------------------------------
print("\nÉTAPE 6 - la méthode ne dépend pas du tokenizer :")
print(f"         mots : {N} tokens -> {N - T} fenêtres de {T} tokens (4 mots, environ 20 caractères)")
print("         BPE ou caractères : plus de tokens, donc plus de fenêtres, mais chacune")
print("         couvre moins de texte : T tokens ne veulent pas dire T mots.")
