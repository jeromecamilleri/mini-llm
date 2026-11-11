"""06_corpus — Leçon 4 : repérer les livres suspects.

Avec 353 livres, on ne peut pas tout lire. On cherche les anomalies avec
les chiffres de metadata.csv (leçon 3) :
  - trop court : extraction ratée, extrait gratuit, livre d'images ;
  - langue : estimée autre que le français, ou différente de celle déclarée ;
  - doublon : même titre et même auteur ;
  - vocabulaire anormal : une diversité très basse ou très haute.

Le script PROPOSE, il ne supprime rien : tu regardes le début du texte de
chaque suspect et tu décides. Les livres écartés vont dans
datasets/exclusions.txt (un nom de fichier par ligne, à remplir à la main).

Prérequis : avoir lancé 03_metadonnees.py (colonne `fichier`).
"""

import csv
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
METADONNEES = RACINE / "datasets" / "metadata.csv"
NETTOYE = RACINE / "datasets" / "cleaned"
SORTIE = RACINE / "datasets" / "a_verifier.csv"

MOTS_MIN = 5000  # en dessous : probablement un extrait ou une extraction ratée
DIVERSITE_MIN, DIVERSITE_MAX = 0.03, 0.5  # mots différents / mots (les livres courts sont plus « divers »)

with open(METADONNEES, encoding="utf-8") as f:
    livres = list(csv.DictReader(f))

suspects = defaultdict(list)  # fichier -> raisons

for l in livres:
    mots, differents = int(l["mots"]), int(l["mots_differents"])
    if mots < MOTS_MIN:
        suspects[l["fichier"]].append(f"court ({mots} mots)")
    if l["langue_estimee"] != "fr":
        suspects[l["fichier"]].append("langue estimée : " + l["langue_estimee"])
    elif not l["langue_declaree"].lower().startswith("fr"):
        suspects[l["fichier"]].append("langue déclarée : " + l["langue_declaree"])
    if mots >= MOTS_MIN:
        diversite = differents / mots
        if not DIVERSITE_MIN <= diversite <= DIVERSITE_MAX:
            suspects[l["fichier"]].append(f"diversité {diversite:.0%}")

par_titre = defaultdict(list)
for l in livres:
    par_titre[(l["titre"].strip().lower(), l["auteur"].strip().lower())].append(l["fichier"])
for fichiers in par_titre.values():
    if len(fichiers) > 1:
        for fichier in fichiers:
            suspects[fichier].append("doublon : " + " / ".join(x[:30] for x in fichiers if x != fichier))

print(f"ÉTAPE 1 — {len(suspects)} suspects sur {len(livres)} livres\n")
for fichier, raisons in sorted(suspects.items()):
    debut = (NETTOYE / fichier).read_text(encoding="utf-8")[:150].replace("\n", " ⏎ ")
    print(f"{fichier}\n  raison : {' ; '.join(raisons)}\n  début  : {debut}\n")

with open(SORTIE, "w", newline="", encoding="utf-8") as f:
    ecrivain = csv.writer(f)
    ecrivain.writerow(["fichier", "raisons"])
    for fichier, raisons in sorted(suspects.items()):
        ecrivain.writerow([fichier, " ; ".join(raisons)])
print(f"écrit : {SORTIE.relative_to(RACINE)}")
print("Pour écarter un livre, ajoute son nom de fichier dans datasets/exclusions.txt")
