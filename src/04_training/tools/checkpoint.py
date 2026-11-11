"""Sauvegarde et rechargement d'un modèle (checkpoint).

Détail pas à pas dans scripts/03_checkpoints.py.
"""
from pathlib import Path

import torch

RACINE = Path(__file__).resolve().parents[3]
DOSSIER_CHECKPOINTS = RACINE / "models" / "checkpoints"
DOSSIER_LOGS = RACINE / "logs"


def sauver(chemin, modele, config, etape, perte_val, optimiseur=None):
    """Écrit un checkpoint : les poids + de quoi reconstruire le modèle (config) + où on en est."""
    contenu = {
        "poids": modele.state_dict(),
        "config": config,  # taille_vocab, d, n_tetes, n_blocs, T_max
        "etape": etape,
        "perte_val": perte_val,
    }
    if optimiseur is not None:
        contenu["optimiseur"] = optimiseur.state_dict()  # permet de REPRENDRE l'entraînement
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    torch.save(contenu, chemin)


def charger(chemin, classe_modele):
    """Reconstruit le modèle à partir de la config, puis y remet les poids."""
    contenu = torch.load(chemin)
    modele = classe_modele(**contenu["config"])
    modele.load_state_dict(contenu["poids"])
    return modele, contenu
