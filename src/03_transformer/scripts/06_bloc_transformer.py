"""
LEÇON 6 (transformer) : le bloc Transformer
===========================================

    .venv/bin/python src/03_transformer/scripts/06_bloc_transformer.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
L'ESSENTIEL
Un bloc = (1) attention : les mots s'échangent des infos ; (2) feed forward :
chaque mot traite seul ce qu'il a reçu. Un GPT = des blocs EMPILÉS (12 pour
GPT-2) : même structure, mais chacun a ses PROPRES poids, appris séparément.
La connexion résiduelle et la LayerNorm sont des sécurités pour l'entraînement.

Leçons 1 à 5 : on sait faire parler les tokens entre eux (attention, masque,
Q/K/V, positions, plusieurs têtes). Mais l'attention ne fait que MÉLANGER des
vecteurs. Il manque de quoi RÉFLÉCHIR sur ce mélange. Le BLOC TRANSFORMER
assemble quatre ingrédients :

  1. ATTENTION (leçons 1 à 5)   : les tokens se parlent.
  2. FEED FORWARD               : un petit réseau classique (01_fundamentals),
                                  appliqué à CHAQUE token SÉPARÉMENT : il réfléchit seul.
  3. CONNEXION RÉSIDUELLE       : on ADDITIONNE le résultat à l'entrée, au lieu de la remplacer.
  4. LAYERNORM (normalisation)  : on remet chaque token à une échelle raisonnable.

SCHÉMA D'UN BLOC (variante de GPT-2)
      X = X + attention( layer_norm(X) )       # les tokens se parlent
      X = X + feed_forward( layer_norm(X) )    # chaque token réfléchit seul
Entrée et sortie ont la MÊME forme (T, d) : on peut empiler autant de blocs qu'on veut.

Paramètres tirés au hasard (comme au début d'un entraînement) : on regarde la
MÉCANIQUE, pas un résultat appris. Dimension 4, 2 têtes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from attention import attention_multi_tetes
from transformer_block import bloc_transformer, feed_forward, gelu, init_bloc, layer_norm

mots = ["le", "chat", "dort", "."]
#                animal action article ponct
X = np.array([
    [0, 0, 1, 0],  # le
    [1, 0, 0, 0],  # chat
    [0, 1, 0, 0],  # dort
    [0, 0, 0, 1],  # .
], dtype=float)
np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(0)
p = init_bloc(4, 2, rng)

# ---------------------------------------------------------------
# ÉTAPE 1 : LayerNorm, chaque token remis à la même échelle
# ---------------------------------------------------------------
petit, grand = np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[10.0, 20.0, 30.0, 40.0]])
print("ÉTAPE 1 - LayerNorm : moyenne 0 et écart-type 1 pour chaque token")
print("         [1, 2, 3, 4]     ->", layer_norm(petit)[0])
print("         [10, 20, 30, 40] ->", layer_norm(grand)[0])
n = layer_norm(grand)
print(f"         moyenne = {n.mean():.2f}, écart-type = {n.std():.2f}")
# Deux vecteurs d'échelles très différentes donnent le même résultat. Dans un
# réseau profond, les nombres peuvent grossir ou rétrécir à chaque couche ;
# la normalisation les garde dans une plage où l'entraînement reste stable
# (idée proche de choisir une bonne initialisation, 01_fundamentals).
# Chaque token est normalisé SEUL (sur ses d nombres), sans regarder les autres.

# ---------------------------------------------------------------
# ÉTAPE 2 : feed forward, chaque token réfléchit seul
# ---------------------------------------------------------------
W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
print("\nÉTAPE 2 - feed forward : d -> 4d -> d, avec GELU au milieu")
print(f"         formes : W1 {W1.shape}, W2 {W2.shape} ; paramètres : {W1.size + b1.size + W2.size + b2.size}")
print("         GELU(-2), GELU(0), GELU(2) =", gelu(np.array([-2.0, 0.0, 2.0])))
# GELU ressemble à ReLU mais sans coin : lisse, et un peu négative près de 0.
# C'est l'activation des Transformers (le « tanh » de 01_fundamentals ne servait
# que pour le petit XOR).

# Test : on change le PREMIER token et on regarde ce qui bouge chez les autres.
X2 = X.copy()
X2[0] = [5.0, -3.0, 2.0, 1.0]
ff, ff2 = feed_forward(X, W1, b1, W2, b2), feed_forward(X2, W1, b1, W2, b2)
_, att = attention_multi_tetes(layer_norm(X), p["tetes"], p["Wo"])
_, att2 = attention_multi_tetes(layer_norm(X2), p["tetes"], p["Wo"])
print("\n         on change le 1er token ; les sorties des tokens 2 à 4 changent-elles ?")
print("           attention    :", not np.allclose(att[1:], att2[1:]), "(les tokens se parlent)")
print("           feed forward :", not np.allclose(ff[1:], ff2[1:]), "(chaque token est traité seul)")

# ---------------------------------------------------------------
# ÉTAPE 3 : la connexion résiduelle, on ajoute au lieu de remplacer
# ---------------------------------------------------------------
_, a = attention_multi_tetes(layer_norm(X), p["tetes"], p["Wo"])
print("\nÉTAPE 3 - connexion résiduelle : sortie = entrée + correction")
print("         entrée de « dort »          :", X[2])
print("         correction (attention)      :", a[2])
print("         entrée + correction         :", X[2] + a[2])
sans = dict(p, W2=np.zeros_like(p["W2"]), Wo=np.zeros_like(p["Wo"]))
print("         si la sous-couche vaut 0, le bloc rend l'entrée telle quelle :",
      np.allclose(bloc_transformer(X, sans), X))
# Deux avantages. (1) Le bloc part de « ne rien changer » et apprend seulement
# des corrections : plus facile à entraîner. (2) La rétropropagation (leçon 5 de
# 01_fundamentals) dispose d'un chemin direct qui traverse l'addition sans être
# affaibli : on peut empiler beaucoup de blocs sans que le gradient s'éteigne.

# ---------------------------------------------------------------
# ÉTAPE 4 : un bloc complet, puis plusieurs empilés
# ---------------------------------------------------------------
Y = bloc_transformer(X, p)
print("\nÉTAPE 4 - un bloc : forme", X.shape, "->", Y.shape)
for m, avant, apres in zip(mots, X, Y):
    print(f"         {m:>5s} : {avant} -> {apres}")
Z = X
for _ in range(3):
    Z = bloc_transformer(Z, init_bloc(4, 2, rng))
print("         3 blocs empilés (paramètres différents) : forme", Z.shape)
# Même forme en entrée et en sortie : l'empilement est possible. Chaque bloc
# affine la représentation : les premiers captent des liens simples, les
# suivants des liens plus abstraits.

# ---------------------------------------------------------------
# ÉTAPE 5 : où sont les paramètres ? (ordre de grandeur de GPT-2)
# ---------------------------------------------------------------
d, n_blocs, vocab, T_max = 768, 12, 50257, 1024
attention_p = 4 * d * d          # Wq, Wk, Wv (toutes têtes) + Wo
ffn_p = 2 * d * 4 * d            # W1 et W2
par_bloc = attention_p + ffn_p
blocs = n_blocs * par_bloc
emb = vocab * d + T_max * d
print(f"\nÉTAPE 5 - GPT-2 (d = {d}, {n_blocs} blocs) :")
print(f"         attention par bloc : {attention_p:,}".replace(",", " "))
print(f"         feed forward par bloc : {ffn_p:,}  (2 fois plus que l'attention)".replace(",", " "))
print(f"         {n_blocs} blocs : {blocs:,}".replace(",", " "))
print(f"         embeddings (tokens + positions) : {emb:,}".replace(",", " "))
print(f"         total : {blocs + emb:,} (environ 124 millions annoncés)".replace(",", " "))
# On retrouve le « 124 millions » de GPT-2. Les deux tiers des paramètres d'un bloc
# sont dans le feed forward : c'est là que le modèle « mémorise » beaucoup de savoir.
