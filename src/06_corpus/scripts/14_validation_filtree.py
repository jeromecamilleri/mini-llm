"""06_corpus — Leçon 14 : A et D sur une validation filtrée (le test est-il équitable ?).

Le problème. Pour l'expérience D, le texte d'entraînement est filtré (leçon 11) mais
la validation ne l'est pas (leçon 12). Elle contient encore un peu de code web,
d'ISBN, de caractères rares. A a appris à prédire ce bruit, pas D : sur ces
quelques tokens, D est très surpris (perte de 8 à 10 au lieu de presque 0). Comme
la perte est une moyenne sur tous les tokens, quelques tokens très mal prédits
suffisent à creuser un écart d'environ 0,03 à 0,04.

L'hypothèse : l'écart entre les courbes de A et de D vient de ce bruit dans la
validation, pas d'un moins bon apprentissage du français. Pour la tester, on mesure
les deux modèles sur DEUX validations :
  - validation.bin     : les 17 livres de validation non filtrés (celle de l'entraînement) ;
  - validation_v2.bin  : les mêmes 17 livres, filtrés comme l'entraînement de D.
Si l'écart A - D disparaît (ou s'inverse) sur la validation filtrée, l'hypothèse tient.

Attention : D n'a un checkpoint `meilleur` que de son étape en cours. L'écart D - A n'a
de sens qu'une fois D terminé. L'effet du bruit sur CHAQUE modèle (perte non filtrée
moins perte filtrée) se lit, lui, à n'importe quelle étape.

Précisions de méthode :
  - on prend 800 fenêtres de T tokens régulièrement espacées sur toute la validation
    (et non 40 batchs tirés au hasard) : mesure déterministe, identique pour les deux modèles ;
  - le calcul se fait sur le CPU, sur 4 fils, pour ne pas gêner un entraînement en cours ;
  - on ne réentraîne rien : on lit les checkpoints `*_meilleur.pt`.

Écrit : datasets/tokenized/validation_v2.bin (ignoré par git).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

CHAPITRE = Path(__file__).resolve().parents[1]
RACINE = CHAPITRE.parents[1]
sys.path.insert(0, str(CHAPITRE / "tools"))
sys.path.insert(0, str(CHAPITRE.parent / "04_training" / "tools"))
from bpe_rapide import BPE  # noqa: E402
from checkpoint import DOSSIER_CHECKPOINTS  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE, charger_bin  # noqa: E402
from gpt_torch import GPT, perte  # noqa: E402

N_FUSIONS = 4000
MODELES = ["livres_A_d384_meilleur.pt", "livres_D_filtre_meilleur.pt"]
TAILLE_BATCH = 32
N_FENETRES = 800  # fenêtres réparties sur toute la validation (tout parcourir prendrait plus d'une heure sur le CPU)
torch.set_num_threads(4)

print("ÉTAPE 1 — validation filtrée (mêmes 17 livres, texte de cleaned_v2)")
chemin_v2 = DOSSIER_TOKENISE / "validation_v2.bin"
if not chemin_v2.exists():
    bpe = BPE.charger(DOSSIER_TOKENISE / f"bpe_{N_FUSIONS}.json")
    noms = (DOSSIER_TOKENISE / "validation_livres.txt").read_text(encoding="utf-8").splitlines()
    livres = [np.array(bpe.encoder((RACINE / "datasets" / "cleaned_v2" / nom).read_text(encoding="utf-8")), dtype=np.uint16)
              for nom in noms]
    np.concatenate(livres).tofile(chemin_v2)
validations = {"non filtrée": charger_bin("validation"), "filtrée": charger_bin("validation_v2")}
for nom, donnees in validations.items():
    print(f"  validation {nom} : {len(donnees):,} tokens")
print(f"  retirés par le filtre : {len(validations['non filtrée']) - len(validations['filtrée']):,} tokens "
      f"({100 * (1 - len(validations['filtrée']) / len(validations['non filtrée'])):.2f} %)")


def perte_moyenne(modele, donnees, T):
    """Perte moyenne sur N_FENETRES fenêtres de T tokens, régulièrement espacées."""
    debuts = np.linspace(0, len(donnees) - T - 2, N_FENETRES).astype(int)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(debuts), TAILLE_BATCH):
            lot = debuts[i : i + TAILLE_BATCH]
            x = torch.from_numpy(np.stack([donnees[d : d + T] for d in lot]).astype(np.int64))
            y = torch.from_numpy(np.stack([donnees[d + 1 : d + T + 1] for d in lot]).astype(np.int64))
            total += perte(modele, x, y).item() * len(lot)
            n += len(lot)
    return total / n


print("\nÉTAPE 2 — perte de chaque modèle sur chaque validation (CPU)")
resultats = {}
for fichier in MODELES:
    contenu = torch.load(DOSSIER_CHECKPOINTS / fichier, map_location="cpu")
    modele = GPT(**contenu["config"])
    modele.load_state_dict(contenu["poids"])
    modele.eval()
    T = contenu["config"]["T_max"]
    for nom, donnees in validations.items():
        debut = time.time()
        resultats[(fichier, nom)] = perte_moyenne(modele, donnees, T)
        print(f"  {fichier:32s} validation {nom:12s} : {resultats[(fichier, nom)]:.4f}  ({time.time() - debut:.0f} s)")
    print(f"    (checkpoint à l'étape {contenu.get('etape', '?')})")

print("\nÉTAPE 3 — bilan")
A, D = MODELES
print("  écart D - A (positif = D moins bon ; à lire seulement quand D est terminé, à 20 000 étapes comme A) :")
for nom in validations:
    print(f"    validation {nom:12s} : D - A = {resultats[(D, nom)] - resultats[(A, nom)]:+.4f}")
print("  effet du bruit sur chaque modèle : perte non filtrée - perte filtrée (valable à toute étape) :")
for fichier in MODELES:
    print(f"    {fichier:32s} : {resultats[(fichier, 'non filtrée')] - resultats[(fichier, 'filtrée')]:+.4f}")
print("  si A est plus bas sur la non filtrée (signe -) et D plus haut (signe +), le bruit avantage bien A.")
