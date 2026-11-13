"""Environnement Sudoku pour apprentissage par renforcement — sudoku_env.py

Interface :
    env = SudokuEnv(grille_init, solution)
    etat = env.reset()
    etat, recompense, termine, info = env.step(action)

Récompenses (signal binaire, sans PEUT_ETRE) :
    +1.0   chiffre correct (OUI)
    -1.0   chiffre incorrect ou action impossible (NON)
   +10.0   bonus ajouté quand la grille est entièrement résolue (s'additionne)
    0.0    timeout (n_coups_max dépassé, le dernier step retourne 0.0)

Le réseau neuronal (Phase 2) recevra en entrée un tenseur dérivé de l'état ;
l'environnement expose deux formats :
    - grille brute  : np.ndarray (9, 9) int8
    - état encodé   : np.ndarray (9, 9, 10) float32   (one-hot, pour le CNN)

Les cases fixes (données de départ) ne peuvent jamais être modifiées.
"""
import numpy as np
from oracle import Oracle
from sudoku_utils import decode_action, encode_action, est_complete, N_ACTIONS


# ──────────────────────────────────────────────────────────────
# Constantes de récompense (faciles à ajuster en Phase 3)
# ──────────────────────────────────────────────────────────────
R_CORRECT = 1.0       # Oracle dit OUI
R_INCORRECT = -1.0    # Oracle dit NON (doublon, mauvais chiffre, case remplie)
R_VICTOIRE = 10.0     # Bonus de fin d'épisode, ajouté au R_CORRECT du dernier coup
R_TIMEOUT = 0.0       # Fin par épuisement des coups


class SudokuEnv:
    """Environnement Sudoku RL.

    Parameters
    ----------
    grille_init : np.ndarray (9,9) int8
        Grille de départ. Cases vides = 0, cases fixes ≠ 0.
    solution : np.ndarray (9,9) int8
        Solution complète (connue, transmise à l'oracle).
    n_coups_max : int
        Nombre de coups avant terminaison forcée (défaut : 500).
    """

    def __init__(
        self,
        grille_init: np.ndarray,
        solution: np.ndarray,
        n_coups_max: int = 500,
    ):
        if grille_init.shape != (9, 9) or solution.shape != (9, 9):
            raise ValueError("grille_init et solution doivent être (9, 9)")

        self._grille_init = grille_init.astype(np.int8).copy()
        self._oracle = Oracle(solution)
        self.n_coups_max = n_coups_max

        # Masque des cases fixes (True = fixe, ne peut pas être modifiée)
        self._fixes = self._grille_init != 0

        # État courant (initialisé par reset)
        self._grille: np.ndarray = None  # type: ignore
        self._n_coups: int = 0
        self._termine: bool = False

        self.reset()

    # ──────────────────────────────────────────────────────────
    # Interface principale
    # ──────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Remet la grille à son état initial. Retourne la grille (9,9)."""
        self._grille = self._grille_init.copy()
        self._n_coups = 0
        self._termine = False
        return self._grille.copy()

    def step(self, action: int) -> tuple:
        """Applique une action et retourne (état, récompense, terminé, info).

        Parameters
        ----------
        action : int dans [0, 728]

        Returns
        -------
        état       : np.ndarray (9,9) int8
        récompense : float
        terminé    : bool
        info       : dict avec 'oracle_verdict', 'n_coups', 'n_cases_restantes'
        """
        if self._termine:
            raise RuntimeError(
                "step() appelé sur un épisode terminé. Appelez reset() d'abord."
            )

        ligne, col, chiffre = decode_action(action)
        verdict = self._oracle.verifier(self._grille, ligne, col, chiffre)

        if verdict == "OUI":
            self._grille[ligne, col] = chiffre
            recompense = R_CORRECT
        else:
            recompense = R_INCORRECT  # NON : on ne modifie pas la grille

        self._n_coups += 1

        # Test de terminaison
        if verdict == "OUI" and est_complete(self._grille):
            recompense += R_VICTOIRE
            self._termine = True
        elif self._n_coups >= self.n_coups_max:
            recompense = R_TIMEOUT
            self._termine = True

        n_restantes = int(np.sum(self._grille == 0))
        info = {
            "oracle_verdict": verdict,
            "n_coups": self._n_coups,
            "n_cases_restantes": n_restantes,
        }

        return self._grille.copy(), recompense, self._termine, info

    # ──────────────────────────────────────────────────────────
    # État encodé pour le réseau (one-hot)
    # ──────────────────────────────────────────────────────────

    def etat_encode(self) -> np.ndarray:
        """Retourne l'état encodé en one-hot : ndarray (9, 9, 10) float32.

        Canal 0 : 1.0 si la case est vide, 0.0 sinon
        Canaux 1–9 : 1.0 si la case contient le chiffre k, 0.0 sinon

        Ce format est directement consommable par un CNN (Phase 2) après
        transposition en (10, 9, 9) ou par un MLP après aplatissement.
        """
        enc = np.zeros((9, 9, 10), dtype=np.float32)
        for r in range(9):
            for c in range(9):
                v = int(self._grille[r, c])
                if v == 0:
                    enc[r, c, 0] = 1.0
                else:
                    enc[r, c, v] = 1.0
        return enc

    def etat_encode_fixe(self) -> np.ndarray:
        """Masque binaire des cases fixes : ndarray (9, 9) float32.

        Utile pour masquer les actions impossibles en Phase 2.
        """
        return self._fixes.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # Propriétés utilitaires
    # ──────────────────────────────────────────────────────────

    @property
    def grille(self) -> np.ndarray:
        return self._grille.copy()

    @property
    def n_coups(self) -> int:
        return self._n_coups

    @property
    def n_cases_vides(self) -> int:
        return int(np.sum(self._grille == 0))

    @property
    def n_cases_initiales_vides(self) -> int:
        """Nombre de cases à remplir depuis le départ."""
        return int(np.sum(self._grille_init == 0))

    @property
    def termine(self) -> bool:
        return self._termine

    def actions_non_fixes(self) -> list:
        """Actions qui ne ciblent pas une case fixe (cases données au départ).

        Note : inclut les cases déjà remplies par le réseau pendant cet épisode.
        C'est l'oracle qui retournera NON si la case est déjà remplie.
        Pour un masquage plus strict (Phase 2), utiliser actions_legales().
        """
        actions = []
        for r in range(9):
            for c in range(9):
                if not self._fixes[r, c]:
                    for d in range(1, 10):
                        actions.append(encode_action(r, c, d))
        return actions

    def actions_legales(self) -> list:
        """Actions sur les cases encore vides (non fixes ET non encore remplies)."""
        actions = []
        for r in range(9):
            for c in range(9):
                if self._grille[r, c] == 0:
                    for d in range(1, 10):
                        actions.append(encode_action(r, c, d))
        return actions

    def render(self) -> str:
        """Représentation ASCII de la grille courante."""
        from sudoku_utils import afficher_grille
        return afficher_grille(self._grille)

    # ──────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SudokuEnv(coups={self._n_coups}/{self.n_coups_max}, "
            f"vides={self.n_cases_vides}, terminé={self._termine})"
        )
