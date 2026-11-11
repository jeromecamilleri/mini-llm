"""Génération de texte avec un modèle entraîné : charger, prédire, choisir, recommencer.

Détail pas à pas dans scripts/01_generation.py et scripts/02_echantillonnage.py.
Réutilise le GPT et les checkpoints de 04_training.
"""
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "04_training" / "tools"))
from checkpoint import DOSSIER_CHECKPOINTS, charger  # noqa: E402
from donnees import charger_corpus  # noqa: E402
from gpt_torch import GPT  # noqa: E402


def charger_modele(nom="animaux_meilleur.pt"):
    """Retourne (tokenizer, modèle en mode évaluation) à partir d'un checkpoint."""
    tok, _, _ = charger_corpus()
    modele, _ = charger(DOSSIER_CHECKPOINTS / nom, GPT)
    modele.eval()
    return tok, modele


@torch.no_grad()
def scores_suivant(modele, ids):
    """Scores (logits) du PROCHAIN mot, pour une liste d'identifiants.

    Le modèle ne voit que les T_max derniers mots (sa fenêtre de contexte) ;
    on ne garde que la dernière position.
    """
    entree = torch.tensor([ids[-modele.T_max:]], device=next(modele.parameters()).device)
    return modele(entree)[0, -1]


def probabilites(scores, temperature=1.0):
    """softmax(scores / température). Température basse : le favori domine."""
    return (scores / temperature).softmax(dim=-1)


def filtrer_top_k(probas, k):
    """Ne garde que les k mots les plus probables, puis renormalise (somme = 1)."""
    garde = probas.topk(k).indices
    masque = torch.zeros_like(probas)
    masque[garde] = 1
    probas = probas * masque
    return probas / probas.sum()


def filtrer_top_p(probas, p):
    """Garde les mots les plus probables jusqu'à cumuler au moins p, puis renormalise."""
    triees, indices = probas.sort(descending=True)
    cumul = triees.cumsum(dim=0)
    # un mot est gardé si le cumul AVANT lui n'a pas encore atteint p (le premier est toujours gardé)
    garde = (cumul - triees) < p
    masque = torch.zeros_like(probas)
    masque[indices[garde]] = 1
    probas = probas * masque
    return probas / probas.sum()


def choisir(probas, generateur=None):
    """Tirage au sort d'un mot selon les probabilités. Sans générateur : le plus probable (glouton)."""
    if generateur is None:
        return int(probas.argmax())
    return int(torch.multinomial(probas, 1, generator=generateur))


def generer(modele, ids, n_nouveaux, temperature=1.0, top_k=None, top_p=None, generateur=None):
    """Génération autorégressive : un mot à la fois, chaque mot choisi est remis en entrée."""
    ids = list(ids)
    for _ in range(n_nouveaux):
        probas = probabilites(scores_suivant(modele, ids), temperature)
        if top_k is not None:
            probas = filtrer_top_k(probas, top_k)
        if top_p is not None:
            probas = filtrer_top_p(probas, top_p)
        ids.append(choisir(probas, generateur))
    return ids
