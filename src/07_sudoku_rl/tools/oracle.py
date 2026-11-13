"""Oracle Sudoku — oracle.py

L'oracle est le seul composant qui connaît la solution.
Il retourne deux verdicts possibles :
  'OUI'  — le chiffre est celui de la solution ET ne crée pas de doublon.
  'NON'  — action impossible (case déjà remplie) OU doublon OU mauvais chiffre.

La réponse 'PEUT_ETRE' (légal mais non vérifié) est supprimée par rapport
au cahier des charges initial : un signal binaire OUI/NON converge bien
plus rapidement pendant l'entraînement par renforcement.

Contrat de pureté :
  - L'oracle ne connaît que la solution. Pas de solver, pas de stratégie.
  - Le réseau ne voit jamais la solution directement.
  - L'oracle est créé une fois par grille (par épisode).
"""
import numpy as np
from sudoku_utils import a_doublon


class Oracle:
    """Oracle binaire lié à une grille.

    Parameters
    ----------
    solution : np.ndarray, shape (9, 9), dtype int8
        Solution complète de la grille en cours.
    """

    def __init__(self, solution: np.ndarray):
        if solution.shape != (9, 9):
            raise ValueError(f"solution doit être (9, 9), reçu {solution.shape}")
        self._solution = solution.copy()

    def verifier(self, grille: np.ndarray, ligne: int, col: int, chiffre: int) -> str:
        """Retourne 'OUI' ou 'NON'.

        Algorithme (ordre de priorité) :
          1. Si la case est déjà remplie → 'NON'  (action impossible)
          2. Si le chiffre crée un doublon ligne/col/bloc → 'NON'
          3. Si chiffre ≠ solution[ligne, col] → 'NON'
          4. Sinon → 'OUI'

        Parameters
        ----------
        grille  : état courant de la grille (cases vides = 0)
        ligne   : index ligne [0, 8]
        col     : index colonne [0, 8]
        chiffre : valeur proposée [1, 9]
        """
        # 1. Case déjà remplie ?
        if grille[ligne, col] != 0:
            return "NON"

        # 2. Doublon ?
        if a_doublon(grille, ligne, col, chiffre):
            return "NON"

        # 3. Bon chiffre ?
        if int(self._solution[ligne, col]) != chiffre:
            return "NON"

        # 4. Tout bon
        return "OUI"

    @property
    def solution(self) -> np.ndarray:
        """Lecture seule de la solution (pour vérifications externes)."""
        return self._solution.copy()
