"""Environnement Sudoku par lots — env_lots.py

Mêmes règles et mêmes récompenses que sudoku_env.SudokuEnv, mais B grilles à la fois,
dans des tenseurs PyTorch (CPU ou GPU), sans boucle Python.

Structure de données (par grille) :
    grille    (9, 9)   chiffre posé (0 = vide)
    solution  (9, 9)   connue de l'oracle seulement
    ligne_a   (9, 9)   ligne_a[l, d]  : le chiffre d+1 est déjà dans la ligne l
    col_a     (9, 9)   col_a[c, d]    : déjà dans la colonne c
    bloc_a    (9, 9)   bloc_a[b, d]   : déjà dans le bloc b  (b = 3 × (l // 3) + c // 3)

Poser un chiffre met à jour trois booléens ; le masque des actions autorisées se déduit
des trois tableaux sans balayer la grille.

Une grille terminée est « gelée » : elle ignore les actions, ne change plus et renvoie 0.
"""
import torch
import torch.nn.functional as F

R_CORRECT = 1.0
R_INCORRECT = -1.0
R_VICTOIRE = 10.0
R_TIMEOUT = 0.0


def _presence(grille: torch.Tensor):
    """Grilles (B, 9, 9) → (ligne_a, col_a, bloc_a), chacun (B, 9, 9) bool."""
    B = grille.shape[0]
    present = F.one_hot(grille, 10)[..., 1:].bool()                  # (B, ligne, col, chiffre)
    ligne_a = present.any(dim=2)
    col_a = present.any(dim=1)
    bloc_a = present.reshape(B, 3, 3, 3, 3, 9).any(dim=(2, 4)).reshape(B, 9, 9)
    return ligne_a, col_a, bloc_a


def refus_en_canaux(refuse: torch.Tensor) -> torch.Tensor:
    """Coups refusés (B, 729) bool, indexés ligne × 81 + colonne × 9 + chiffre − 1 → (B, 9, 9, 9) float32.

    Canal d = « le chiffre d+1 a déjà été refusé dans cette case » (même rangement que les chiffres de l'état).
    """
    return refuse.reshape(-1, 9, 9, 9).permute(0, 3, 1, 2).float()


class SudokuEnvLots:
    def __init__(self, grilles, solutions, n_coups_max: int = 500, appareil="cpu", memoire: bool = False):
        self.n_coups_max = n_coups_max
        self.appareil = appareil
        self.memoire = memoire  # True : un coup refusé est retiré du masque pour cette grille
        self.charger(grilles, solutions)

    def charger(self, grilles, solutions) -> None:
        """Démarre B nouvelles parties à partir de grilles et solutions (B, 9, 9)."""
        self.depart = torch.as_tensor(grilles, dtype=torch.long, device=self.appareil).clone()
        self.solution = torch.as_tensor(solutions, dtype=torch.long, device=self.appareil).clone()
        self.B = self.depart.shape[0]
        self._idx = torch.arange(self.B, device=self.appareil)
        self.reset()

    def reset(self) -> torch.Tensor:
        """Remet toutes les grilles à leur départ. Retourne les grilles (B, 9, 9)."""
        self.grille = self.depart.clone()
        self.ligne_a, self.col_a, self.bloc_a = _presence(self.grille)
        self.n_coups = torch.zeros(self.B, dtype=torch.long, device=self.appareil)
        self.n_vides = (self.grille == 0).sum(dim=(1, 2))
        self.termine = torch.zeros(self.B, dtype=torch.bool, device=self.appareil)
        self.refuse = torch.zeros(self.B, 729, dtype=torch.bool, device=self.appareil)
        return self.grille.clone()

    def recharger(self, indices, grilles, solutions) -> None:
        """Remplace les parties d'indices donnés par de nouvelles grilles (pour enchaîner les parties)."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.appareil)
        g = torch.as_tensor(grilles, dtype=torch.long, device=self.appareil)
        s = torch.as_tensor(solutions, dtype=torch.long, device=self.appareil)
        self.depart[indices] = g
        self.solution[indices] = s
        self.grille[indices] = g
        la, ca, ba = _presence(g)
        self.ligne_a[indices], self.col_a[indices], self.bloc_a[indices] = la, ca, ba
        self.n_coups[indices] = 0
        self.n_vides[indices] = (g == 0).sum(dim=(1, 2))
        self.termine[indices] = False
        self.refuse[indices] = False

    def step(self, actions: torch.Tensor):
        """actions (B,) entiers 0–728. Retourne (grilles, récompenses (B,), terminé (B,), verdicts (B,) bool)."""
        a = actions.to(self.appareil).long()
        l, c, d = a // 81, (a // 9) % 9, a % 9
        b = 3 * (l // 3) + c // 3
        i = self._idx
        actif = ~self.termine

        vide = self.grille[i, l, c] == 0
        doublon = self.ligne_a[i, l, d] | self.col_a[i, c, d] | self.bloc_a[i, b, d]
        bon = self.solution[i, l, c] == d + 1
        oui = actif & vide & ~doublon & bon

        recompense = torch.where(oui, R_CORRECT, R_INCORRECT)
        o = oui.nonzero(as_tuple=True)[0]
        self.grille[o, l[o], c[o]] = d[o] + 1
        self.ligne_a[o, l[o], d[o]] = True
        self.col_a[o, c[o], d[o]] = True
        self.bloc_a[o, b[o], d[o]] = True
        m = (actif & ~oui).nonzero(as_tuple=True)[0]
        self.refuse[m, a[m]] = True  # toujours noté ; il ne change le masque que si memoire=True

        self.n_coups += actif.long()
        self.n_vides -= oui.long()
        victoire = oui & (self.n_vides == 0)
        timeout = actif & ~victoire & (self.n_coups >= self.n_coups_max)
        recompense = recompense + R_VICTOIRE * victoire
        recompense = torch.where(timeout, R_TIMEOUT, recompense)
        recompense = torch.where(actif, recompense, 0.0)
        self.termine |= victoire | timeout
        return self.grille.clone(), recompense, self.termine.clone(), oui

    def masques(self) -> torch.Tensor:
        """Actions autorisées (B, 729) : case vide et chiffre absent de la ligne, colonne, bloc."""
        bloc = self.bloc_a.reshape(self.B, 3, 3, 9).repeat_interleave(3, 1).repeat_interleave(3, 2)
        interdit = (self.ligne_a[:, :, None, :] | self.col_a[:, None, :, :] | bloc
                    | (self.grille != 0)[..., None])
        autorises = (~interdit).reshape(self.B, 729)
        return autorises & ~self.refuse if self.memoire else autorises

    def etats(self, avec_refus: bool = False) -> torch.Tensor:
        """Entrée du réseau : one-hot (B, 10, 9, 9) float32 ; avec_refus : (B, 19, 9, 9) (voir refus_en_canaux)."""
        etats = F.one_hot(self.grille, 10).permute(0, 3, 1, 2).float()
        return torch.cat([etats, refus_en_canaux(self.refuse)], dim=1) if avec_refus else etats
