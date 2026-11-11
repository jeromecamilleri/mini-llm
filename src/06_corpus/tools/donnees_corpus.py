"""Lecture des corpus tokenisés (datasets/tokenized/*.bin) et tirage de batchs."""

from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[3]
DOSSIER_TOKENISE = RACINE / "datasets" / "tokenized"


def charger_bin(nom):
    """Lit train.bin ou validation.bin : un long tableau d'ids (uint16)."""
    return np.fromfile(DOSSIER_TOKENISE / f"{nom}.bin", dtype=np.uint16)


def tirer_batch(donnees, taille_batch, T, rng, appareil="cpu"):
    """Tire taille_batch fenêtres de T tokens au hasard.

    x = les T tokens, y = les mêmes décalés d'un cran : pour chaque position,
    la « bonne réponse » est le token suivant.
    """
    debuts = rng.integers(0, len(donnees) - T - 1, size=taille_batch)
    x = np.stack([donnees[d : d + T] for d in debuts]).astype(np.int64)
    y = np.stack([donnees[d + 1 : d + T + 1] for d in debuts]).astype(np.int64)
    return torch.from_numpy(x).to(appareil), torch.from_numpy(y).to(appareil)
