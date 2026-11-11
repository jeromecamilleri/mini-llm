"""Préparation des données pour l'entraînement : texte -> identifiants -> batchs (x, y).

Même principe que 02_tokenizer/scripts/05_sequences.py, regroupé pour être réutilisé.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_tokenizer" / "tools"))
from tokenizer_basic import TokenizerMots

CORPUS = Path(__file__).resolve().parents[2] / "02_tokenizer" / "corpus_animaux.txt"


def charger_corpus(part_entrainement=0.9):
    """Retourne (tokenizer, ids d'entraînement, ids de validation).

    On coupe le TEXTE en deux blocs avant de fabriquer les fenêtres, pour qu'aucune
    fenêtre de validation n'ait été vue pendant l'entraînement.
    """
    texte = CORPUS.read_text(encoding="utf-8").strip()
    tok = TokenizerMots(texte)
    ids = torch.tensor(tok.encoder(texte))
    coupure = int(part_entrainement * len(ids))
    return tok, ids[:coupure], ids[coupure:]


def tirer_batch(donnees, taille_batch, T, generateur):
    """taille_batch fenêtres prises au hasard : x = ids[i:i+T], y = décalé d'un cran."""
    debuts = torch.randint(0, len(donnees) - T, (taille_batch,), generator=generateur)
    x = torch.stack([donnees[d:d + T] for d in debuts])
    y = torch.stack([donnees[d + 1:d + T + 1] for d in debuts])
    return x, y
