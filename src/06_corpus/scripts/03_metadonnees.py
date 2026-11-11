"""06_corpus — Leçon 3 : métadonnées et statistiques par livre.

Avec 353 livres, on veut savoir ce qu'on a : qui, quelle langue, quelle
taille. Sans ça, impossible de comparer des corpus (expériences A à D :
français / technique / mixte) ni de repérer un livre cassé.

Pour chaque livre on écrit une ligne dans datasets/metadata.csv :
  - lues dans le fichier Kindle (titre, auteur, éditeur, langue) ;
  - mesurées sur le texte nettoyé (caractères, mots, mots différents...) ;
  - une langue estimée par nous-mêmes, pour vérifier celle déclarée.
"""

import csv
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mobi import lire_metadonnees  # noqa: E402

ORIGINAUX = RACINE / "datasets" / "originals"
NETTOYE = RACINE / "datasets" / "cleaned"
SORTIE = RACINE / "datasets" / "metadata.csv"

# Mots très fréquents : un texte français en contient beaucoup « le », « de »...
MOTS_FR = {"le", "la", "les", "de", "des", "et", "un", "une", "est", "que", "qui", "dans", "il"}
MOTS_EN = {"the", "of", "and", "to", "in", "is", "that", "it", "was", "he", "with"}


def langue_estimee(mots):
    fr = sum(m in MOTS_FR for m in mots)
    en = sum(m in MOTS_EN for m in mots)
    return "fr" if fr > en else "en"


def mots_du_texte(texte):
    return re.findall(r"\w+(?:'\w+)?", texte.lower())


lignes = []
for fichier in sorted(NETTOYE.glob("*.txt")):
    source = next(p for p in ORIGINAUX.iterdir() if p.stem == fichier.stem)
    meta = lire_metadonnees(source)
    texte = fichier.read_text(encoding="utf-8")
    mots = mots_du_texte(texte)
    lignes.append({
        "id": fichier.stem[:40],
        "fichier": fichier.name,
        "titre": meta.get("titre", ""),
        "auteur": meta.get("auteur", ""),
        "editeur": meta.get("editeur", ""),
        "format": source.suffix[1:],
        "langue_declaree": meta.get("langue", ""),
        "langue_estimee": langue_estimee(mots),
        "caracteres": len(texte),
        "mots": len(mots),
        "mots_differents": len(set(mots)),
        "paragraphes": texte.count("\n") + 1,
        "lettres_par_mot": round(sum(map(len, mots)) / len(mots), 2),
    })

print("ÉTAPE 1 — la table")
with open(SORTIE, "w", newline="", encoding="utf-8") as f:
    ecrivain = csv.DictWriter(f, fieldnames=list(lignes[0]))
    ecrivain.writeheader()
    ecrivain.writerows(lignes)
print(f"  écrit : {SORTIE.relative_to(RACINE)} ({len(lignes)} livres)\n")

print(f"  {'titre':<24}{'auteur':<22}{'langue':<8}{'mots':>8}{'différents':>12}{'diversité':>11}")
for l in lignes:
    diversite = l["mots_differents"] / l["mots"]
    print(f"  {l['titre'][:22]:<24}{l['auteur'][:20]:<22}{l['langue_declaree'] + '/' + l['langue_estimee']:<8}"
          f"{l['mots']:>8,}{l['mots_differents']:>12,}{diversite:>11.0%}")

print("\nÉTAPE 2 — total du corpus")
total = sum(l["mots"] for l in lignes)
print(f"  {total:,} mots, {sum(l['caracteres'] for l in lignes):,} caractères")
print("  langues déclarées != estimées :", [l["titre"] for l in lignes if l["langue_declaree"] != l["langue_estimee"]])
