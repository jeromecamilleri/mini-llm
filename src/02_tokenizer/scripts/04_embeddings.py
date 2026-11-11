"""
LEÇON 4 (tokenizer) : les embeddings, un vecteur appris par token
=================================================================

    .venv/bin/python src/02_tokenizer/scripts/04_embeddings.py

PLACE DANS LE PARCOURS (chapitre 02_tokenizer)
Leçon 1 : un identifiant n'a pas de sens (« 3 » n'est pas « proche » de « 2 »),
et le one-hot met tous les tokens à égale distance.
Ici : l'EMBEDDING, un petit vecteur de nombres PAR TOKEN, qui est APPRIS pendant
l'entraînement. Les tokens qui s'utilisent de la même façon finissent avec des
vecteurs proches : le sens apparaît tout seul.

IDÉE CENTRALE
  - Une table E de forme (taille du vocabulaire, dimension) : une ligne par token.
  - Embedding du token i = la ligne i de E. C'est exactement « one-hot @ E ».
  - E est un paramètre comme W1 ou b1 (leçons 9 à 11) : on la corrige par rétropropagation.

TÂCHE D'ENTRAÎNEMENT : prédire le mot SUIVANT (c'est aussi celle d'un GPT).
Perte = -log(probabilité donnée au vrai mot suivant) : 0 si le modèle est
certain et a raison, grande s'il est sûr de lui et se trompe (« cross entropy »,
vue en détail dans 04_training).
Dimension 2 pour pouvoir TOUT dessiner (un GPT en utilise des centaines).

Corpus : `corpus_animaux.txt` (dossier 02_tokenizer), tokenisé en mots (leçon 2).

ET LE XOR (dossier 01_fundamentals) DANS TOUT ÇA ?
(on le reprendra dans 04_training, quand on entraînera le vrai mini-GPT)
  - Adam (leçon 8) : on utilisera AdamW, un Adam avec une meilleure régularisation.
  - Rétropropagation / autograd, boucle d'entraînement, validation, nn.Module,
    DataLoader (leçons 5, 6, 9, 10, 11) : repris tels quels pour le mini-GPT.
  - Cross entropy : la perte vue ici sera celle du mini-GPT.
  - tanh : en général remplacée par GELU dans un Transformer (le verdict « tanh
    gagne » valait pour le XOR). Le principe reste : comparer en ne changeant
    qu'une variable.
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # accès aux modules
from tokenizer_basic import TokenizerMots

corpus = (Path(__file__).resolve().parents[1] / "corpus_animaux.txt").read_text(encoding="utf-8").strip()
tok = TokenizerMots(corpus)
mots = tok.vocab.id_vers_token
ids = torch.tensor(tok.encoder(corpus))
V = len(mots)
rang = tok.vocab.token_vers_id


def cosinus(E, a, b):
    """Ressemblance de DIRECTION entre deux vecteurs : 1 = même, 0 = sans rapport, -1 = opposés."""
    va, vb = E[rang[a]], E[rang[b]]
    return (va @ vb / (va.norm() * vb.norm())).item()


# ---------------------------------------------------------------
# ÉTAPE 1 : un embedding est une ligne d'une table
# ---------------------------------------------------------------
torch.manual_seed(0)
emb = nn.Embedding(V, 2)  # table (13, 2), tirée au hasard
chat = rang["chat"]
une_ligne = emb.weight[chat]
one_hot = torch.eye(V)[chat]
print("ÉTAPE 1 - table E :", tuple(emb.weight.shape), "(un vecteur de 2 nombres par token)")
print("         embedding de 'chat' =", une_ligne.detach().numpy().round(3))
print("         one-hot de 'chat' @ E =", (one_hot @ emb.weight).detach().numpy().round(3))
# Même résultat : prendre la ligne i, c'est multiplier le one-hot par E.
# Mais prendre une ligne est bien plus rapide que multiplier un grand vecteur creux.


# ---------------------------------------------------------------
# ÉTAPE 2 : avant l'entraînement, aucun sens
# ---------------------------------------------------------------
E0 = emb.weight.detach().clone()
paires = [("chat", "chien"), ("chat", "oiseau"), ("dort", "mange"), ("chat", "dort")]
print("\nÉTAPE 2 - ressemblance (cosinus) AVANT entraînement :")
for a, b in paires:
    print(f"         {a:6s} ~ {b:6s} : {cosinus(E0, a, b):+.2f}")
# Des vecteurs tirés au hasard : les valeurs n'ont aucune signification.


# ---------------------------------------------------------------
# ÉTAPE 3 : entraîner l'embedding à prédire le mot suivant
# ---------------------------------------------------------------
# Entrée : un mot. Cible : le mot qui le suit dans le corpus.
# Modèle : embedding (2 nombres) -> une couche linéaire -> un score par mot du vocabulaire.
x, y = ids[:-1], ids[1:]
sortie = nn.Linear(2, V)
opt = torch.optim.Adam(list(emb.parameters()) + list(sortie.parameters()), lr=0.05)
print("\nÉTAPE 3 - entraînement (prédire le mot suivant)")
for epoque in range(1, 601):
    opt.zero_grad()
    perte = nn.functional.cross_entropy(sortie(emb(x)), y)
    perte.backward()
    opt.step()
    if epoque in (1, 50, 150, 600):
        print(f"         époque {epoque:3d} : perte = {perte.item():.3f}")
# La perte ne tombe pas à 0 : après « chat », le mot suivant peut être « dort »,
# « mange » ou « court ». Le meilleur modèle ne peut qu'estimer ces probabilités.


# ---------------------------------------------------------------
# ÉTAPE 4 : après l'entraînement, du sens est apparu
# ---------------------------------------------------------------
E = emb.weight.detach()
print("\nÉTAPE 4 - ressemblance (cosinus) APRÈS entraînement :")
for a, b in paires:
    print(f"         {a:6s} ~ {b:6s} : {cosinus(E, a, b):+.2f}   (avant : {cosinus(E0, a, b):+.2f})")
voisins = sorted((m for m in mots if m != "chat"), key=lambda m: -cosinus(E, "chat", m))
print("         mots les plus proches de 'chat' :", voisins[:4])
# Personne n'a dit au modèle que « chat », « chien », « oiseau » sont des animaux.
# Il a seulement constaté qu'ils sont suivis des mêmes mots (dort, mange, court) :
# leurs vecteurs se sont rapprochés. Idem pour les verbes, suivis de « . » ou « aussi ».


# ---------------------------------------------------------------
# ÉTAPE 5 : les vecteurs sont des paramètres comme les autres
# ---------------------------------------------------------------
n_emb = emb.weight.numel()
n_sortie = sum(p.numel() for p in sortie.parameters())
print(f"\nÉTAPE 5 - paramètres : embedding {n_emb} + couche de sortie {n_sortie} = {n_emb + n_sortie}")
# Pour un GPT : vocabulaire V x dimension d. Avec V = 50 000 et d = 768 (GPT-2),
# l'embedding seul compte déjà 38 millions de nombres. D'où l'intérêt d'un
# vocabulaire de taille raisonnable (leçon 3) pour la GTX 1070.
