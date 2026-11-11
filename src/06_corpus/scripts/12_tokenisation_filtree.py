"""06_corpus — Leçon 12 : tokeniser le texte filtré pour l'entraînement (expérience D).

On repart des livres filtrés de la leçon 11 (datasets/cleaned_v2/) et on écrit
datasets/tokenized/train_v2.bin.

Ce qui ne change PAS, pour que la comparaison reste propre :
  - le BPE (bpe_4000.json) : même vocabulaire, mêmes ids ;
  - les livres de validation (validation_livres.txt) : ce sont les mêmes ;
  - validation.bin : il n'est pas réécrit. Il contient les livres NON filtrés,
    donc la perte de validation reste mesurée sur le même texte que pour la
    référence et pour B.

Seul le texte d'entraînement change : c'est la seule variable de l'expérience D.
Dans la leçon 7, il suffit alors de mettre DONNEES_TRAIN = "train_v2".
"""

import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bpe_rapide import BPE  # noqa: E402

FILTRE = RACINE / "datasets" / "cleaned_v2"
SORTIE = RACINE / "datasets" / "tokenized"
N_FUSIONS = 4000

bpe = BPE.charger(SORTIE / f"bpe_{N_FUSIONS}.json")
validation = set((SORTIE / "validation_livres.txt").read_text(encoding="utf-8").splitlines())
fichiers = [f for f in sorted(FILTRE.glob("*.txt")) if f.name not in validation]

print(f"ÉTAPE 1 — {len(fichiers)} livres d'entraînement (les {len(validation)} de validation sont ignorés)")

print("\nÉTAPE 2 — tokeniser")
debut = time.time()
livres, inconnus = [], 0
for numero, f in enumerate(fichiers, 1):
    livre = np.array(bpe.encoder(f.read_text(encoding="utf-8")), dtype=np.uint16)
    inconnus += int((livre == 0).sum())
    livres.append(livre)
    if numero % 50 == 0:
        print(f"  {numero}/{len(fichiers)} livres ({time.time() - debut:.0f} s)")

print("\nÉTAPE 3 — écrire train_v2.bin")
tout = np.concatenate(livres)
tout.tofile(SORTIE / "train_v2.bin")
ancien = (SORTIE / "train.bin").stat().st_size // 2
print(f"  train_v2.bin : {len(tout):,} tokens (train.bin : {ancien:,}, {100 * (ancien - len(tout)) / ancien:.2f} % de moins)")
print(f"  tokens inconnus (<inc>) : {inconnus:,} ({inconnus / len(tout):.4%})")
print("  début décodé :", repr(bpe.decoder(tout[:60].tolist())))
