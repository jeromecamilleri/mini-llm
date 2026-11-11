"""06_corpus — Leçon 17 : tokeniser le théâtre et réserver des dialogues de validation.

Après la leçon 16 : `datasets/claire/theatre_propre.txt` (5,02 M de tokens). Il faut
maintenant le transformer en fichiers `.bin` utilisables par l'entraînement, et
décider ce qui sert à apprendre et ce qui sert à mesurer.

POURQUOI UNE VALIDATION DE DIALOGUES. La perte de validation actuelle se mesure sur
17 livres. Un modèle qui apprend à dialoguer mieux ne le montrerait presque pas sur
des romans. On réserve donc une partie du théâtre qui n'est jamais apprise, et on
la mesure à part.

COMMENT ON COUPE. Les pièces se suivent dans le fichier. On prend la FIN du fichier
(environ 5 % des tokens) et on coupe sur une ligne vide (changement de scène) :
une scène n'est donc jamais partagée entre apprentissage et validation. Une pièce
entière, ou presque, est mise de côté : le modèle n'a jamais vu ses personnages ni
ses répliques. Découper au hasard réplique par réplique serait faux : les répliques
voisines se ressemblent, la validation serait trop facile.

Fichiers écrits dans `datasets/tokenized/` (dossier ignoré par git) :
  theatre_train.bin       le théâtre d'apprentissage
  theatre_validation.bin  les dialogues de validation (jamais appris)
  train_theatre.bin       train.bin (livres) + theatre_train.bin : l'option « mélange simple »

L'entraînement tire des fenêtres au hasard dans tout le tableau (`tirer_batch`) : mettre
les deux corpus bout à bout suffit, le modèle voit les livres et le théâtre mélangés
dans chaque batch, en proportion de leur taille (environ 7 % de théâtre).

Le BPE ne change pas : une seule variable change (le corpus).

Lancement : .venv/bin/python src/06_corpus/scripts/17_tokenisation_theatre.py
"""

import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bpe_rapide import BPE  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE, charger_bin  # noqa: E402

ENTREE = RACINE / "datasets" / "claire" / "theatre_propre.txt"
BPE_JSON = DOSSIER_TOKENISE / "bpe_4000.json"
PART_VALIDATION = 0.05  # fraction du fichier réservée à la validation

print("ÉTAPE 1 — lire le théâtre propre")
lignes = ENTREE.read_text(encoding="utf-8").split("\n")
print(f"  {len(lignes):,} lignes")

print("\nÉTAPE 2 — choisir la coupure (sur une ligne vide, vers la fin)")
cible = int(len(lignes) * (1 - PART_VALIDATION))
coupure = next(i for i in range(cible, len(lignes)) if lignes[i] == "")
train_texte = "\n".join(lignes[:coupure]) + "\n"
val_texte = "\n".join(lignes[coupure + 1 :]).rstrip("\n") + "\n"
print(f"  coupure à la ligne {coupure:,} sur {len(lignes):,} (visée : {cible:,})")
print(f"  dernière réplique d'apprentissage : {lignes[coupure - 1][:80]!r}")
print(f"  première réplique de validation   : {lignes[coupure + 1][:80]!r}")

print("\nÉTAPE 3 — tokeniser (BPE existant)")
bpe = BPE.charger(BPE_JSON)
ids_train = np.array(bpe.encoder(train_texte), dtype=np.uint16)
ids_val = np.array(bpe.encoder(val_texte), dtype=np.uint16)
total = len(ids_train) + len(ids_val)
print(f"  théâtre d'apprentissage : {len(ids_train):,} tokens ({len(ids_train) / total:.1%})")
print(f"  dialogues de validation : {len(ids_val):,} tokens ({len(ids_val) / total:.1%})")

print("\nÉTAPE 4 — écrire les fichiers")
ids_train.tofile(DOSSIER_TOKENISE / "theatre_train.bin")
ids_val.tofile(DOSSIER_TOKENISE / "theatre_validation.bin")
livres = charger_bin("train")
melange = np.concatenate([livres, ids_train])
melange.tofile(DOSSIER_TOKENISE / "train_theatre.bin")
print(f"  theatre_train.bin       : {len(ids_train):,} tokens")
print(f"  theatre_validation.bin  : {len(ids_val):,} tokens")
print(f"  train_theatre.bin       : {len(melange):,} tokens = {len(livres):,} livres + {len(ids_train):,} théâtre")
print(f"  part du théâtre dans le mélange : {len(ids_train) / len(melange):.1%}")

print("\nÉTAPE 5 — vérification")
relu = np.fromfile(DOSSIER_TOKENISE / "theatre_validation.bin", dtype=np.uint16)
print("  validation relue :", len(relu), "tokens ; ids de", relu.min(), "à", relu.max())
print("  début décodé :", repr(bpe.decoder(relu[:40].tolist())))
