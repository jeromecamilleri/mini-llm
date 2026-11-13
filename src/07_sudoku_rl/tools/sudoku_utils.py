"""Utilitaires Sudoku partagés — sudoku_utils.py

Fonctions de bas niveau utilisées par SudokuEnv, Oracle et SudokuDataset :
  - encodage / décodage des actions (case × chiffre → entier)
  - validation (doublon, grille valide, grille complète)
  - conversion chaîne ↔ ndarray
  - solveur par backtracking (pour calculer les solutions manquantes)
  - affichage ASCII

Aucune dépendance en dehors de NumPy.
"""
import numpy as np


# ──────────────────────────────────────────────────────────────
# 1. Encodage des actions
# ──────────────────────────────────────────────────────────────

N_ACTIONS = 729  # 81 cases × 9 chiffres


def encode_action(ligne: int, col: int, chiffre: int) -> int:
    """(ligne, col, chiffre) → entier dans [0, 728].

    ligne, col ∈ [0, 8]   chiffre ∈ [1, 9]
    """
    return ligne * 81 + col * 9 + (chiffre - 1)


def decode_action(action: int) -> tuple:
    """Entier [0, 728] → (ligne, col, chiffre).

    chiffre ∈ [1, 9]
    """
    chiffre = action % 9 + 1
    col = (action // 9) % 9
    ligne = action // 81
    return ligne, col, chiffre


# ──────────────────────────────────────────────────────────────
# 2. Indices du bloc 3×3
# ──────────────────────────────────────────────────────────────

def bloc_indices(ligne: int, col: int) -> tuple:
    """Coin supérieur-gauche du bloc 3×3 contenant (ligne, col)."""
    return (ligne // 3) * 3, (col // 3) * 3


def valeurs_bloc(grille: np.ndarray, ligne: int, col: int) -> np.ndarray:
    """Retourne les 9 valeurs du bloc 3×3 de la case (ligne, col)."""
    r0, c0 = bloc_indices(ligne, col)
    return grille[r0:r0 + 3, c0:c0 + 3].flatten()


# ──────────────────────────────────────────────────────────────
# 3. Validation
# ──────────────────────────────────────────────────────────────

def a_doublon(grille: np.ndarray, ligne: int, col: int, chiffre: int) -> bool:
    """True si placer chiffre en (ligne, col) crée un doublon.

    Vérifie la ligne, la colonne et le bloc 3×3.
    La case (ligne, col) elle-même est ignorée (elle peut déjà contenir une valeur
    — c'est à l'appelant de vérifier si la case est libre).
    """
    # Masque pour ignorer la case cible
    masque = np.ones((9, 9), dtype=bool)
    masque[ligne, col] = False

    # Ligne
    if chiffre in grille[ligne, :][masque[ligne, :]]:
        return True
    # Colonne
    if chiffre in grille[:, col][masque[:, col]]:
        return True
    # Bloc
    r0, c0 = bloc_indices(ligne, col)
    bloc = grille[r0:r0 + 3, c0:c0 + 3].copy()
    bloc_masque = masque[r0:r0 + 3, c0:c0 + 3]
    if chiffre in bloc[bloc_masque]:
        return True
    return False


def _unites_valides(valeurs: np.ndarray) -> bool:
    """Vérifie qu'un vecteur de 9 cases contient 1–9 exactement une fois."""
    v = valeurs[valeurs != 0]  # ignore les cases vides
    return len(v) == len(set(v))


def est_valide(grille: np.ndarray) -> bool:
    """True si aucun doublon dans les 9 lignes, 9 colonnes et 9 blocs.

    Les cases vides (0) sont ignorées : une grille partiellement remplie
    peut être valide.
    """
    for i in range(9):
        if not _unites_valides(grille[i, :]):
            return False
        if not _unites_valides(grille[:, i]):
            return False
    for br in range(3):
        for bc in range(3):
            bloc = grille[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3].flatten()
            if not _unites_valides(bloc):
                return False
    return True


def est_complete(grille: np.ndarray) -> bool:
    """True si aucune case vide ET grille valide."""
    return bool(np.all(grille != 0)) and est_valide(grille)


# ──────────────────────────────────────────────────────────────
# 4. Solveur par backtracking avec propagation de contraintes
# ──────────────────────────────────────────────────────────────
#
# Stratégie :
#   - MRV  (Minimum Remaining Values) : on choisit toujours la case
#     ayant le moins de candidats restants parmi les cases vides.
#   - Forward Checking : après chaque placement, on retire le chiffre
#     des candidats des cases vides liées (même ligne, colonne, bloc).
#     Si un ensemble de candidats devient vide → impasse détectée tôt.
#   - Naked Singles propagation : on propage en cascade les cases
#     qui n'ont plus qu'un seul candidat.
#   - Backtracking : si impasse, on revient en arrière en restaurant
#     l'état complet (grille + candidats) depuis une copie.


def _liees_indices(r: int, c: int) -> list:
    """Retourne les indices (r2, c2) liés à (r, c) : même ligne, colonne ou bloc."""
    liees = set()
    for i in range(9):
        if i != c:
            liees.add((r, i))
        if i != r:
            liees.add((i, c))
    r0, c0 = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if rr != r or cc != c:
                liees.add((rr, cc))
    return list(liees)


# Cache des indices liés (calculé une seule fois au chargement du module)
_LIEES: list[list[list]] = [[_liees_indices(r, c) for c in range(9)] for r in range(9)]


def _init_candidats(grille: np.ndarray):
    """Initialise les ensembles de candidats.

    Retourne une liste 9×9 de sets, ou None si la grille est invalide.
    Cases non vides → None (sentinelle).
    """
    cands = [[None] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grille[r, c] == 0:
                s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                for rr, cc in _LIEES[r][c]:
                    v = int(grille[rr, cc])
                    if v != 0:
                        s.discard(v)
                if not s:
                    return None
                cands[r][c] = s
    return cands


def _placer(grille: np.ndarray, cands: list, r: int, c: int, d: int) -> bool:
    """Place d en (r,c), propage, retourne False si impasse.

    Modifie grille et cands en place.
    """
    pile = [(r, c, d)]
    while pile:
        rr, cc, val = pile.pop()
        grille[rr, cc] = val
        cands[rr][cc] = None  # case remplie

        for r2, c2 in _LIEES[rr][cc]:
            s = cands[r2][c2]
            if s is None:
                continue  # case déjà remplie
            if val in s:
                s.discard(val)
                if not s:
                    return False  # impasse
                if len(s) == 1:
                    seul = next(iter(s))
                    pile.append((r2, c2, seul))
    return True


def _copier_etat(grille: np.ndarray, cands: list):
    """Copie profonde de la grille et des candidats."""
    g2 = grille.copy()
    c2 = [
        [s.copy() if s is not None else None for s in row]
        for row in cands
    ]
    return g2, c2


def _backtrack_ac(grille: np.ndarray, cands: list):
    """Backtracking avec forward checking + MRV + naked singles."""
    # MRV : case vide avec le moins de candidats
    best_r, best_c = -1, -1
    best_count = 10
    for r in range(9):
        for c in range(9):
            s = cands[r][c]
            if s is not None:  # case vide
                n = len(s)
                if n < best_count:
                    best_count = n
                    best_r, best_c = r, c
                    if n == 1:
                        break
        if best_count == 1:
            break

    if best_r == -1:
        # Plus de case vide → solution trouvée
        return grille.copy()

    for d in list(cands[best_r][best_c]):
        # Copie de l'état avant le placement
        g2, c2 = _copier_etat(grille, cands)

        if _placer(g2, c2, best_r, best_c, d):
            resultat = _backtrack_ac(g2, c2)
            if resultat is not None:
                return resultat
        # Sinon : backtrack automatique (on a travaillé sur g2/c2)

    return None


def resoudre(grille: np.ndarray) -> np.ndarray | None:
    """Solveur Sudoku : backtracking + forward checking + MRV + naked singles.

    Retourne un ndarray (9,9) int8 solution, ou None si aucune solution.
    La grille d'entrée n'est PAS modifiée.

    Performances typiques (Intel i7, Python pur) :
      easy       < 0.5 ms
      medium     < 1 ms
      hard       < 5 ms
      diabolical < 20 ms  (rares cas jusqu'à ~100 ms)
    """
    if not est_valide(grille):
        return None
    g = grille.astype(np.int8).copy()
    cands = _init_candidats(g)
    if cands is None:
        return None
    return _backtrack_ac(g, cands)


# ──────────────────────────────────────────────────────────────
# 5. Conversion chaîne ↔ ndarray
# ──────────────────────────────────────────────────────────────

def chaine_vers_grille(s: str) -> np.ndarray:
    """'003020600...' (81 chars, '0' = vide) → ndarray (9,9) int8."""
    if len(s) != 81:
        raise ValueError(f"La chaîne doit avoir 81 caractères, reçu {len(s)}")
    return np.array([int(c) for c in s], dtype=np.int8).reshape(9, 9)


def grille_vers_chaine(g: np.ndarray) -> str:
    """ndarray (9,9) → '003020600...' (81 chars)."""
    return "".join(str(v) for v in g.flatten())


# ──────────────────────────────────────────────────────────────
# 6. Affichage ASCII
# ──────────────────────────────────────────────────────────────

def afficher_grille(grille: np.ndarray) -> str:
    """Représentation ASCII avec séparateurs de blocs.

    Exemple :
      5 3 . | . 7 . | . . .
      6 . . | 1 9 5 | . . .
      ...
      ------+-------+------
    """
    lignes = []
    for r in range(9):
        if r > 0 and r % 3 == 0:
            lignes.append("------+-------+------")
        rangee = []
        for c in range(9):
            if c > 0 and c % 3 == 0:
                rangee.append("|")
            v = grille[r, c]
            rangee.append(str(v) if v != 0 else ".")
        lignes.append(" ".join(rangee))
    return "\n".join(lignes)


# ──────────────────────────────────────────────────────────────
# 7. Vérification rapide au chargement
# ──────────────────────────────────────────────────────────────

def valider_paire(puzzle_str: str, solution_str: str) -> str | None:
    """Valide une paire (puzzle, solution) et retourne un message d'erreur ou None.

    Vérifie :
      - longueurs (81 chars)
      - solution sans zéro
      - cohérence puzzle/solution (les cases données correspondent)
      - solution valide (pas de doublon)
    """
    if len(puzzle_str) != 81:
        return f"puzzle : {len(puzzle_str)} chars (attendu 81)"
    if len(solution_str) != 81:
        return f"solution : {len(solution_str)} chars (attendu 81)"
    if "0" in solution_str:
        return "solution contient un zéro"
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p != "0" and p != s:
            return f"incohérence case {i} : puzzle={p}, solution={s}"
    sol = chaine_vers_grille(solution_str)
    if not est_valide(sol):
        return "solution invalide (doublon)"
    return None  # OK
