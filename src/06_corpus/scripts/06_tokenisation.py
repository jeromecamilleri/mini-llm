"""06_corpus — Leçon 6 : tokeniser tout le corpus et garder des livres pour la validation.

Le modèle ne lit pas du texte : il lit des suites de numéros (les « ids » des
tokens). On transforme donc chaque livre avec le BPE de la leçon 5.

Deux fichiers sont produits dans datasets/tokenized/ :
  - train.bin       : les livres d'entraînement, mis bout à bout ;
  - validation.bin  : des livres ENTIERS mis de côté, jamais vus à l'entraînement.

Pourquoi des livres entiers ? Si on prenait des morceaux au hasard dans tous les
livres, le modèle verrait pendant l'entraînement les pages voisines de celles de
la validation (mêmes personnages, mêmes phrases) : la validation paraîtrait trop
bonne. Un livre jamais vu mesure vraiment ce que le modèle sait généraliser
(rappel : leçon 2 de 04_training, le surapprentissage).

Format : chaque id est écrit sur 2 octets (uint16, valeurs de 0 à 65 535 : notre
vocabulaire de ~4 600 tokens y tient). 70 millions de tokens = environ 140 Mo.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bpe_rapide import BPE  # noqa: E402

NETTOYE = RACINE / "datasets" / "cleaned"
EXCLUSIONS = RACINE / "datasets" / "exclusions.txt"
SORTIE = RACINE / "datasets" / "tokenized"
N_FUSIONS = 4000
PART_VALIDATION = 0.05  # part des livres mis de côté
GRAINE = 0  # fixe : on retrouve toujours les mêmes livres de validation
MAX_LIVRES = None  # un entier pour tester vite

bpe = BPE.charger(SORTIE / f"bpe_{N_FUSIONS}.json")
exclus = set(EXCLUSIONS.read_text(encoding="utf-8").splitlines()) if EXCLUSIONS.exists() else set()
fichiers = [f for f in sorted(NETTOYE.glob("*.txt")) if f.name not in exclus][:MAX_LIVRES]

print("ÉTAPE 1 — choisir les livres de validation")
n_val = max(1, round(len(fichiers) * PART_VALIDATION))
validation = set(random.Random(GRAINE).sample(fichiers, n_val))
print(f"  {len(fichiers) - n_val} livres d'entraînement, {n_val} de validation :")
for f in sorted(validation):
    print("   -", f.name[:70])
(SORTIE / "validation_livres.txt").write_text("\n".join(f.name for f in sorted(validation)) + "\n", encoding="utf-8")

print("\nÉTAPE 2 — tokeniser")
debut = time.time()
ids = {"train": [], "validation": []}
inconnus = 0
for numero, f in enumerate(fichiers, 1):
    livre = np.array(bpe.encoder(f.read_text(encoding="utf-8")), dtype=np.uint16)
    inconnus += int((livre == 0).sum())
    ids["validation" if f in validation else "train"].append(livre)
    if numero % 50 == 0:
        print(f"  {numero}/{len(fichiers)} livres ({time.time() - debut:.0f} s)")

print("\nÉTAPE 3 — écrire les fichiers")
for nom, livres in ids.items():
    tout = np.concatenate(livres)
    tout.tofile(SORTIE / f"{nom}.bin")
    print(f"  {nom}.bin : {len(tout):,} tokens ({tout.nbytes / 1e6:.0f} Mo)")
total = sum(len(l) for livres in ids.values() for l in livres)
print(f"  tokens inconnus (<inc>) : {inconnus:,} sur {total:,} ({inconnus / total:.4%})")

print("\nÉTAPE 4 — vérification")
train = np.fromfile(SORTIE / "train.bin", dtype=np.uint16)
print("  relu depuis le disque :", len(train), "tokens ; ids de", train.min(), "à", train.max())
print("  début décodé :", repr(bpe.decoder(train[:60].tolist())))
