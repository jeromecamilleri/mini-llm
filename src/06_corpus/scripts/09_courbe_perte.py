"""06_corpus — Leçon 9 : tracer la courbe d'apprentissage.

    .venv/bin/python src/06_corpus/scripts/09_courbe_perte.py

On relit logs/entrainement_livres.csv (écrit par la leçon 7 : étape, perte train,
perte validation, taux d'apprentissage) et on trace deux graphiques :

  1. La perte d'entraînement et de validation. Elle descend vite au début, puis
     de plus en plus lentement (rendements décroissants). Si la validation
     remontait alors que la perte train continue de descendre, ce serait du
     surapprentissage : le modèle apprendrait les livres par cœur.
  2. Le taux d'apprentissage : échauffement, puis descente en cosinus.

L'écart entre les deux courbes de perte indique si le modèle est limité par sa
taille (courbes proches, comme ici) ou par le manque de données (écart qui grandit).

Écrit : logs/courbe_entrainement_livres.png
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RACINE = Path(__file__).resolve().parents[3]
CSV = RACINE / "logs" / "entrainement_livres.csv"
IMAGE = RACINE / "logs" / "courbe_entrainement_livres.png"

with open(CSV, newline="") as f:
    lignes = list(csv.DictReader(f))
etapes = [int(l["etape"]) for l in lignes]
train = [float(l["perte_train"]) for l in lignes]
val = [float(l["perte_val"]) for l in lignes]
taux = [float(l["taux"]) for l in lignes]

print(f"{len(etapes)} mesures, de l'étape {etapes[0]} à {etapes[-1]}")
print(f"validation : {val[0]:.3f} -> {val[-1]:.3f} (perplexité {math.exp(val[0]):.0f} -> {math.exp(val[-1]):.0f})")
print(f"écart validation - train à la fin : {val[-1] - train[-1]:.3f}")
gains = [val[i - 1] - val[i] for i in range(1, len(val))]
print(f"gain de validation par tranche : {gains[0]:.3f} au début, {gains[-1]:.3f} à la fin")

fig, (haut, bas) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
haut.plot(etapes, train, label="entraînement")
haut.plot(etapes, val, label="validation")
haut.axhline(math.log(4612), color="gray", linestyle=":", label="hasard : ln(4612) = 8,44")
haut.set_ylabel("perte")
haut.set_ylim(top=6)
haut.legend()
haut.set_title("Perte (plus bas = meilleur)")
bas.plot(etapes, taux, color="tab:green")
bas.set_ylabel("taux d'apprentissage")
bas.set_xlabel("étape")
bas.set_title("Taux d'apprentissage : échauffement puis cosinus")
fig.tight_layout()
fig.savefig(IMAGE, dpi=120)
print(f"écrit : {IMAGE.relative_to(RACINE)}")
