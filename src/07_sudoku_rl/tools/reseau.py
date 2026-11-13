"""Réseau Sudoku (CNN, tête politique + tête valeur) — reseau.py

Même mécanique que 04_training/tools/gpt_torch.py (nn.Module, forward, backward),
avec des convolutions à la place de l'attention.

Entrée  : état one-hot (B, 10, 9, 9) float32 (canal 0 = vide, canaux 1 à 9 = chiffre).
Sorties : logits de politique (B, 729) et valeur (B,).

Action = ligne × 81 + colonne × 9 + (chiffre − 1), comme encode_action.
La politique sort donc d'abord en (B, 9 chiffres, 9, 9) puis est réordonnée en (ligne, colonne, chiffre).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def encoder_etats(grilles) -> torch.Tensor:
    """Grilles (B, 9, 9) d'entiers 0–9 → états one-hot (B, 10, 9, 9) float32.

    Version vectorisée de SudokuEnv.etat_encode (qui boucle en Python).
    """
    g = torch.as_tensor(np.asarray(grilles), dtype=torch.long)
    return F.one_hot(g, 10).permute(0, 3, 1, 2).float()


def masques_actions(grilles: torch.Tensor) -> torch.Tensor:
    """Grilles (B, 9, 9) → masques booléens (B, 729) : True si l'action est autorisée.

    Autorisée = case vide ET chiffre absent de la ligne, de la colonne et du bloc 3×3.
    Version vectorisée de evaluation.masque_actions, utilisable sur GPU.
    """
    B = grilles.shape[0]
    present = F.one_hot(grilles.long(), 10)[..., 1:].bool()          # (B, ligne, col, chiffre)
    dans_ligne = present.any(dim=2)[:, :, None, :]                   # (B, ligne, 1, chiffre)
    dans_col = present.any(dim=1)[:, None, :, :]                     # (B, 1, col, chiffre)
    blocs = present.reshape(B, 3, 3, 3, 3, 9).any(dim=(2, 4))        # (B, bloc_l, bloc_c, chiffre)
    dans_bloc = blocs.repeat_interleave(3, dim=1).repeat_interleave(3, dim=2)
    rempli = (grilles != 0)[..., None]
    return (~(dans_ligne | dans_col | dans_bloc | rempli)).reshape(B, 729)


class BlocResiduel(nn.Module):
    """Deux convolutions 3×3 et une connexion résiduelle : sortie = entrée + correction."""

    def __init__(self, canaux):
        super().__init__()
        self.conv1 = nn.Conv2d(canaux, canaux, 3, padding=1)
        self.conv2 = nn.Conv2d(canaux, canaux, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class ReseauSudoku(nn.Module):
    def __init__(self, canaux=64, n_blocs=4, voit_refus=False):
        """voit_refus=True : 9 canaux d'entrée en plus (19 au total) montrent les coups déjà refusés
        sur cette grille : canal 10 + d = « le chiffre d+1 a déjà été refusé dans cette case »."""
        super().__init__()
        self.voit_refus = voit_refus
        self.entree = nn.Conv2d(19 if voit_refus else 10, canaux, 3, padding=1)
        self.blocs = nn.Sequential(*[BlocResiduel(canaux) for _ in range(n_blocs)])
        self.tete_politique = nn.Conv2d(canaux, 9, 1)  # un score par chiffre et par case
        self.tete_valeur = nn.Linear(canaux, 1)
        # Dernière couche de la politique à zéro : au départ, tous les scores sont égaux,
        # donc la politique masquée est exactement uniforme (= l'agent « hasard + masque »).
        nn.init.zeros_(self.tete_politique.weight)
        nn.init.zeros_(self.tete_politique.bias)

    def forward(self, etats, masques=None):
        """etats : (B, 10, 9, 9). Retourne (logits (B, 729), valeur (B,)).

        Si masques (B, 729) est donné, les actions interdites reçoivent un score de −1e9
        (probabilité nulle après softmax).
        """
        h = self.blocs(F.relu(self.entree(etats)))
        logits = self.tete_politique(h).permute(0, 2, 3, 1).reshape(-1, 729)
        valeur = self.tete_valeur(h.mean(dim=(2, 3))).squeeze(-1)
        if masques is not None:
            logits = logits.masked_fill(~masques, -1e9)
        return logits, valeur

    def nb_parametres(self):
        return sum(p.numel() for p in self.parameters())


def charger_modele(chemin, appareil="cpu") -> "ReseauSudoku":
    """Recrée un réseau à partir d'un checkpoint : 19 canaux d'entrée = le réseau voit les refus."""
    etat = torch.load(chemin, map_location=appareil)
    modele = ReseauSudoku(voit_refus=etat["entree.weight"].shape[1] == 19).to(appareil)
    modele.load_state_dict(etat)
    return modele


def politique_reseau(modele, appareil="cpu", glouton=False):
    """Fabrique une politique (env, rng) -> action à partir du réseau, pour evaluation.evaluer.

    Tire l'action selon les probabilités du réseau (ou la plus probable si glouton).
    """
    def politique(env, rng):
        grille = torch.as_tensor(env.grille, dtype=torch.long)[None].to(appareil)
        with torch.no_grad():
            logits, _ = modele(encoder_etats(grille.cpu()).to(appareil), masques_actions(grille))
        if glouton:
            return int(logits.argmax(dim=1))
        probas = F.softmax(logits[0], dim=0).cpu().numpy().astype(np.float64)
        probas /= probas.sum()
        return int(rng.choice(729, p=probas))
    return politique
