"""Dataset Sudoku — dataset.py

Charge les fichiers du dépôt sudoku-exchange-puzzle-bank (format : hash puzzle rating)
et les normalise vers un format CSV interne avec solutions calculées.

Usage rapide :
    from dataset import SudokuDataset
    ds = SudokuDataset("datasets/sudoku/sudoku_dataset.csv")
    item = ds[0]
    # item = {'puzzle': ndarray(9,9), 'solution': ndarray(9,9),
    #         'difficulte': 'facile', 'rating': 1.2, 'id': 0}

    facile = ds.par_difficulte('facile')
    ds_mel = ds.melanger(graine=42)

Construction du CSV depuis les fichiers bruts :
    Voir scripts/02_dataset.py qui appelle construire_dataset().
"""
import csv
import sys
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

# Chemin vers tools/ pour les imports internes
sys.path.insert(0, str(Path(__file__).parent))
from sudoku_utils import (
    chaine_vers_grille,
    grille_vers_chaine,
    resoudre,
    valider_paire,
)

# ──────────────────────────────────────────────────────────────
# Format CSV interne
# ──────────────────────────────────────────────────────────────
# puzzle,solution,difficulte,rating
# 003020600...,483921657...,facile,1.2
#
# difficulte : 'facile' | 'moyen' | 'difficile' | 'diabolique'
# rating     : flottant (note originale du dépôt source)

COLONNES = ["puzzle", "solution", "difficulte", "rating"]

# Correspondance fichier source → difficulté
FICHIERS_SOURCE = {
    "easy.txt": "facile",
    "medium.txt": "moyen",
    "hard.txt": "difficile",
    "diabolical.txt": "diabolique",
}


# ──────────────────────────────────────────────────────────────
# Lecture du format source (hash puzzle rating)
# ──────────────────────────────────────────────────────────────

def _lire_fichier_source(chemin: Path) -> Iterator[tuple]:
    """Génère (puzzle_str, rating_float) depuis un fichier brut.

    Format attendu par ligne : '0000183b305c 050703060007... 1.2'
      12 chars hash + espace + 81 chars puzzle + espace + 4 chars rating
    """
    with open(chemin, "r", encoding="ascii") as f:
        for numero, ligne in enumerate(f, 1):
            ligne = ligne.rstrip("\n")
            if len(ligne) < 98:
                continue  # ligne trop courte, ignorée
            puzzle_str = ligne[13:94]   # octets 13–93 (81 chars)
            rating_str = ligne[95:99].strip()
            try:
                rating = float(rating_str)
            except ValueError:
                continue
            if len(puzzle_str) != 81 or not puzzle_str.isdigit():
                continue
            yield puzzle_str, rating


# ──────────────────────────────────────────────────────────────
# Construction du CSV
# ──────────────────────────────────────────────────────────────

def construire_dataset(
    dossier_raw: str | Path,
    chemin_sortie: str | Path,
    n_max_par_niveau: int = 50_000,
    verbose: bool = True,
) -> None:
    """Lit les fichiers bruts, calcule les solutions manquantes et écrit le CSV.

    Parameters
    ----------
    dossier_raw      : dossier contenant easy.txt, medium.txt, hard.txt, diabolical.txt
    chemin_sortie    : chemin du CSV à écrire
    n_max_par_niveau : nombre maximal de grilles à inclure par niveau de difficulté
                       (réduit la taille pour l'entraînement local)
    verbose          : affiche la progression
    """
    dossier_raw = Path(dossier_raw)
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # Les fichiers source contiennent puzzle + rating mais PAS la solution.
    # On doit résoudre chaque puzzle avec notre solveur backtracking.
    # (Les fichiers source garantissent une solution unique par construction.)

    n_total = 0
    n_erreurs = 0
    n_insolubles = 0

    with open(chemin_sortie, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(COLONNES)

        for fichier, difficulte in FICHIERS_SOURCE.items():
            chemin_fichier = dossier_raw / fichier
            if not chemin_fichier.exists():
                if verbose:
                    print(f"  [ABSENT] {fichier} — ignoré")
                continue

            n_niveau = 0
            if verbose:
                print(f"  {fichier} ({difficulte})…", flush=True)

            for puzzle_str, rating in _lire_fichier_source(chemin_fichier):
                if n_niveau >= n_max_par_niveau:
                    break

                # Résoudre le puzzle
                grille = chaine_vers_grille(puzzle_str)
                solution = resoudre(grille)

                if solution is None:
                    n_insolubles += 1
                    continue

                solution_str = grille_vers_chaine(solution)

                # Valider la paire
                erreur = valider_paire(puzzle_str, solution_str)
                if erreur:
                    n_erreurs += 1
                    warnings.warn(f"Paire rejetée ({fichier}) : {erreur}")
                    continue

                writer.writerow([puzzle_str, solution_str, difficulte, f"{rating:.1f}"])
                n_niveau += 1
                n_total += 1

                if verbose and n_niveau % 1000 == 0:
                    print(f"    {n_niveau} grilles…", flush=True)

            if verbose:
                print(f"    → {n_niveau} grilles retenues")

    if verbose:
        print(f"\nDataset écrit : {chemin_sortie}")
        print(f"  Total       : {n_total}")
        print(f"  Insolubles  : {n_insolubles}")
        print(f"  Erreurs val : {n_erreurs}")


def construire_dataset_mini(
    chemin_complet: str | Path,
    chemin_sortie: str | Path,
    n_par_niveau: int = 250,
) -> None:
    """Extrait un mini-dataset depuis le CSV complet (pour dev rapide).

    Prend n_par_niveau grilles de chaque niveau (facile/moyen/difficile/diabolique).
    """
    chemin_complet = Path(chemin_complet)
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    compteurs: dict[str, int] = {}
    with open(chemin_complet, "r", encoding="utf-8") as fin, \
         open(chemin_sortie, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(COLONNES)
        for row in reader:
            d = row["difficulte"]
            if compteurs.get(d, 0) < n_par_niveau:
                writer.writerow([row["puzzle"], row["solution"], row["difficulte"], row["rating"]])
                compteurs[d] = compteurs.get(d, 0) + 1

    total = sum(compteurs.values())
    print(f"Mini-dataset écrit : {chemin_sortie}  ({total} grilles)")
    for d, n in sorted(compteurs.items()):
        print(f"  {d:12s} : {n}")


# ──────────────────────────────────────────────────────────────
# SudokuDataset
# ──────────────────────────────────────────────────────────────

class SudokuDataset:
    """Dataset de grilles Sudoku chargé depuis le CSV interne.

    Parameters
    ----------
    chemin_csv : str | Path
        Chemin vers le CSV normalisé (colonnes : puzzle, solution, difficulte, rating).

    Exemples
    --------
    >>> ds = SudokuDataset("datasets/sudoku/sudoku_dataset.csv")
    >>> len(ds)
    200000
    >>> item = ds[0]
    >>> item['puzzle'].shape
    (9, 9)
    >>> facile = ds.par_difficulte('facile')
    >>> ds2 = ds.melanger(graine=0)
    """

    def __init__(self, chemin_csv: str | Path):
        self._chemin = Path(chemin_csv)
        self._donnees: list[dict] = []
        self._charger()

    def _charger(self) -> None:
        """Lit le CSV et valide chaque ligne."""
        n_rejets = 0
        with open(self._chemin, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                erreur = valider_paire(row["puzzle"], row["solution"])
                if erreur:
                    warnings.warn(f"Ligne {i+2} rejetée : {erreur}")
                    n_rejets += 1
                    continue
                try:
                    rating = float(row.get("rating", 0.0))
                except ValueError:
                    rating = 0.0
                self._donnees.append({
                    "puzzle": chaine_vers_grille(row["puzzle"]),
                    "solution": chaine_vers_grille(row["solution"]),
                    "difficulte": row["difficulte"].strip(),
                    "rating": rating,
                    "id": i,
                })
        if n_rejets > 0:
            warnings.warn(f"{n_rejets} lignes rejetées sur {len(self._donnees) + n_rejets}")

    # ──────────────────────────────────────────────────────────
    # Interface standard
    # ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._donnees)

    def __getitem__(self, idx: int) -> dict:
        """Retourne {'puzzle', 'solution', 'difficulte', 'rating', 'id'}."""
        item = self._donnees[idx]
        # On retourne des copies pour éviter les mutations accidentelles
        return {
            "puzzle": item["puzzle"].copy(),
            "solution": item["solution"].copy(),
            "difficulte": item["difficulte"],
            "rating": item["rating"],
            "id": item["id"],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ──────────────────────────────────────────────────────────
    # Filtrage et réordonnancement
    # ──────────────────────────────────────────────────────────

    def par_difficulte(self, niveau: str) -> "SudokuDataset":
        """Retourne une vue filtrée sur un niveau de difficulté.

        niveau : 'facile', 'moyen', 'difficile' ou 'diabolique'
        """
        sous_ensemble = [d for d in self._donnees if d["difficulte"] == niveau]
        if not sous_ensemble:
            raise ValueError(
                f"Niveau '{niveau}' inconnu ou absent du dataset. "
                f"Niveaux disponibles : {self.niveaux_disponibles()}"
            )
        return SudokuDataset._depuis_liste(sous_ensemble)

    def melanger(self, graine: int = 42) -> "SudokuDataset":
        """Retourne une copie avec les grilles dans un ordre aléatoire reproductible."""
        rng = np.random.default_rng(graine)
        indices = rng.permutation(len(self._donnees)).tolist()
        melange = [self._donnees[i] for i in indices]
        return SudokuDataset._depuis_liste(melange)

    def sous_echantillon(self, n: int, graine: int = 42) -> "SudokuDataset":
        """Retourne un sous-ensemble de n grilles (tirage sans remise)."""
        rng = np.random.default_rng(graine)
        n = min(n, len(self._donnees))
        indices = rng.choice(len(self._donnees), size=n, replace=False).tolist()
        return SudokuDataset._depuis_liste([self._donnees[i] for i in indices])

    # ──────────────────────────────────────────────────────────
    # Statistiques
    # ──────────────────────────────────────────────────────────

    def niveaux_disponibles(self) -> list:
        """Liste des niveaux présents dans le dataset."""
        return sorted({d["difficulte"] for d in self._donnees})

    def distribution(self) -> dict:
        """Nombre de grilles par niveau."""
        dist: dict[str, int] = {}
        for d in self._donnees:
            dist[d["difficulte"]] = dist.get(d["difficulte"], 0) + 1
        return dict(sorted(dist.items()))

    def stats_cases_vides(self) -> dict:
        """Min/max/moyenne de cases vides par niveau."""
        par_niveau: dict[str, list] = {}
        for d in self._donnees:
            n_vides = int(np.sum(d["puzzle"] == 0))
            par_niveau.setdefault(d["difficulte"], []).append(n_vides)
        return {
            niv: {
                "min": int(np.min(vals)),
                "max": int(np.max(vals)),
                "mean": float(np.mean(vals)),
                "n": len(vals),
            }
            for niv, vals in sorted(par_niveau.items())
        }

    # ──────────────────────────────────────────────────────────
    # Constructeur interne
    # ──────────────────────────────────────────────────────────

    @classmethod
    def _depuis_liste(cls, donnees: list) -> "SudokuDataset":
        """Construit un SudokuDataset depuis une liste déjà parsée (sans CSV)."""
        obj = object.__new__(cls)
        obj._chemin = Path("<mémoire>")
        obj._donnees = donnees
        return obj

    def __repr__(self) -> str:
        return (
            f"SudokuDataset({len(self)} grilles, "
            f"niveaux={self.niveaux_disponibles()})"
        )
