"""
LEÇON 2 (sudoku_rl) : construire et explorer le dataset
========================================================

    .venv/bin/python src/07_sudoku_rl/scripts/02_dataset.py

PLACE DANS LE PARCOURS
En 04_training, le corpus était un fichier texte découpé en fenêtres.
Ici, les « données » sont des grilles Sudoku avec leur solution.
Le dataset sert uniquement à fournir des problèmes à l'environnement ;
le réseau ne voit jamais les solutions directement (c'est l'oracle qui les consulte).

CE QU'ON VOIT ICI
  Étape 1 : vérifier les fichiers bruts téléchargés
  Étape 2 : construire le dataset CSV complet (avec solveur)
  Étape 3 : construire le mini-dataset (dev rapide, ~1 000 grilles)
  Étape 4 : charger et explorer SudokuDataset
  Étape 5 : distribution par difficulté et stats des cases vides

DURÉE ESTIMÉE
  Construction complète (200 000 grilles) : ~15–30 min selon CPU
  Mini-dataset (1 000 grilles)           : ~1 min
  Passe --mini pour construire uniquement le mini-dataset.

USAGE
    .venv/bin/python src/07_sudoku_rl/scripts/02_dataset.py         # complet
    .venv/bin/python src/07_sudoku_rl/scripts/02_dataset.py --mini  # rapide
"""
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

from dataset import SudokuDataset, construire_dataset, construire_dataset_mini

# Chemins
DOSSIER_RAW     = RACINE / "datasets" / "sudoku" / "raw"
CSV_COMPLET     = RACINE / "datasets" / "sudoku" / "sudoku_dataset.csv"
CSV_MINI        = RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv"

MODE_MINI = "--mini" in sys.argv

# ──────────────────────────────────────────────────────────────
print("=" * 55)
print("ÉTAPE 1 : vérification des fichiers bruts")
print("=" * 55)

fichiers = ["easy.txt", "medium.txt", "hard.txt", "diabolical.txt"]
for f in fichiers:
    p = DOSSIER_RAW / f
    if p.exists():
        taille_mb = p.stat().st_size / 1_000_000
        print(f"  ✓ {f:20s} {taille_mb:.1f} Mo")
    else:
        print(f"  ✗ {f:20s} ABSENT (télécharger depuis sudoku-exchange-puzzle-bank)")

print()

# ──────────────────────────────────────────────────────────────
print("=" * 55)
print("ÉTAPE 2 : construction du dataset CSV complet")
print("=" * 55)

if CSV_COMPLET.exists() and not MODE_MINI:
    print(f"  Déjà présent : {CSV_COMPLET}")
    print("  Supprimer le fichier pour reconstruire.")
elif MODE_MINI and CSV_COMPLET.exists():
    print(f"  Mode --mini : on saute la construction complète (CSV déjà présent)")
elif MODE_MINI and not CSV_COMPLET.exists():
    print("  Mode --mini ET CSV absent : on construit quand même un CSV partiel")
    print("  (50 grilles par niveau pour avoir un CSV de référence)")
    t0 = time.time()
    construire_dataset(DOSSIER_RAW, CSV_COMPLET, n_max_par_niveau=50, verbose=True)
    print(f"  Durée : {time.time() - t0:.1f} s")
else:
    print("  Construction en cours… (patience, le solveur tourne pour chaque grille)")
    print(f"  Sortie : {CSV_COMPLET}")
    # 50 000 grilles par niveau → ~200 000 au total
    # Réduire n_max_par_niveau si la mémoire ou le temps manquent
    t0 = time.time()
    construire_dataset(DOSSIER_RAW, CSV_COMPLET, n_max_par_niveau=50_000, verbose=True)
    duree = time.time() - t0
    print(f"  Durée totale : {duree/60:.1f} min")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 3 : construction du mini-dataset")
print("=" * 55)

if CSV_MINI.exists():
    print(f"  Déjà présent : {CSV_MINI}")
else:
    t0 = time.time()
    construire_dataset_mini(CSV_COMPLET, CSV_MINI, n_par_niveau=250)
    print(f"  Durée : {time.time() - t0:.1f} s")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 4 : charger et explorer SudokuDataset")
print("=" * 55)

# On utilise le mini pour l'exploration (rapide)
csv_a_explorer = CSV_MINI if CSV_MINI.exists() else CSV_COMPLET
print(f"  Chargement de {csv_a_explorer.name}…")
ds = SudokuDataset(csv_a_explorer)
print(f"  {ds}")

item = ds[0]
print(f"\n  Première grille :")
print(f"    difficulte : {item['difficulte']}")
print(f"    rating     : {item['rating']}")
print(f"    puzzle     : {item['puzzle'].shape} int8")
print(f"    solution   : {item['solution'].shape} int8")

from sudoku_utils import afficher_grille
print("\n  Puzzle :")
print(afficher_grille(item["puzzle"]))
print("\n  Solution :")
print(afficher_grille(item["solution"]))

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ÉTAPE 5 : distribution et stats des cases vides")
print("=" * 55)

print("\n  Distribution :")
for niveau, n in ds.distribution().items():
    print(f"    {niveau:12s} : {n:>6} grilles")

print("\n  Cases vides par niveau (critère de difficulté) :")
stats = ds.stats_cases_vides()
for niveau, s in stats.items():
    print(f"    {niveau:12s} : min={s['min']}  max={s['max']}  moy={s['mean']:.1f}  ({s['n']} grilles)")

# Vérification : filtrage par difficulté
print("\n  Test filtrage par_difficulte('facile') :")
facile = ds.par_difficulte("facile")
print(f"    → {len(facile)} grilles")

# Vérification : mélange reproductible
ds1 = ds.melanger(graine=0)
ds2 = ds.melanger(graine=0)
assert ds1[0]["id"] == ds2[0]["id"], "Le mélange n'est pas reproductible !"
print(f"\n  ✓ melanger(graine=0) est reproductible (premier id : {ds1[0]['id']})")

print("\n✓ Leçon 2 terminée.")
