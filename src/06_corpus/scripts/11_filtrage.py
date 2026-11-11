"""06_corpus — Leçon 11 : filtrer le bruit du corpus (expérience D).

Dans les textes générés, on voit des mots inventés, des tables des matières
recopiées. Le modèle apprend ce qu'on lui montre : si des lignes de code ou des
mentions d'éditeur sont dans le corpus, il en apprend autant que de romans.
Cette leçon MESURE d'abord le bruit, puis écrit une copie filtrée des livres.

Vocabulaire :
  - « bruit » : du texte qui n'est pas de la langue qu'on veut apprendre
    (code de feuille de style, ISBN, adresses web, lignes sans lettres...) ;
  - « alphabet » : l'ensemble des caractères conservés. Un caractère très rare
    (idéogramme, grec ancien) n'apparaît pas assez pour être appris ; il ne fait
    que gaspiller de la place dans le vocabulaire.

Règles de tools/filtrage.py, appliquées ligne par ligne :
  code web, édition, code isolé, peu de lettres, alphabet rare.

Ce qui est écrit :
  - datasets/cleaned_v2/ : les livres filtrés. datasets/cleaned/ n'est JAMAIS modifié.

Règle d'une expérience propre : on change UNE variable. Ici seul le texte
d'ENTRAÎNEMENT sera filtré (leçon 12). La validation reste celle d'avant, avec
les mêmes livres non filtrés : ainsi la perte de validation reste comparable à
celle de la référence et de B.
"""

import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from filtrage import alphabet_frequent, filtrer, regle_supprimant  # noqa: E402

NETTOYE = RACINE / "datasets" / "cleaned"
SORTIE = RACINE / "datasets" / "cleaned_v2"
EXCLUSIONS = RACINE / "datasets" / "exclusions.txt"
SEUIL_ALPHABET = 300  # un caractère vu moins de 300 fois sur 220 millions est jugé rare
N_EXEMPLES = 4

exclus = set(EXCLUSIONS.read_text(encoding="utf-8").splitlines())
fichiers = [f for f in sorted(NETTOYE.glob("*.txt")) if f.name not in exclus]
textes = {f.name: f.read_text(encoding="utf-8") for f in fichiers}
total = sum(len(t) for t in textes.values())

print("ÉTAPE 1 — l'alphabet à garder")
alphabet = alphabet_frequent(textes.values(), SEUIL_ALPHABET)
diff = Counter("".join(textes.values()))
print(f"  {len(diff)} caractères différents dans le corpus, {len(alphabet)} gardés (vus >= {SEUIL_ALPHABET} fois)")
retires = sorted((c for c in diff if c not in alphabet), key=lambda c: -diff[c])
print(f"  les plus fréquents des retirés : {[(c, diff[c]) for c in retires[:12]]}")

print("\nÉTAPE 2 — des exemples de lignes supprimées, règle par règle")
vus = Counter()
for nom, t in textes.items():
    for ligne in t.split("\n"):
        regle = regle_supprimant(ligne)
        if regle and vus[regle] < N_EXEMPLES:
            vus[regle] += 1
            print(f"  [{regle}] {nom[:25]} : {ligne[:90]!r}")

print("\nÉTAPE 3 — filtrer tous les livres")
SORTIE.mkdir(exist_ok=True)
compteurs = Counter()
total_filtre = 0
for nom, t in textes.items():
    propre = filtrer(t, alphabet, compteurs)
    (SORTIE / nom).write_text(propre, encoding="utf-8")
    total_filtre += len(propre)
for nom, n in compteurs.most_common():
    print(f"  {nom:<36} {n:>8,} lignes")

print("\nÉTAPE 4 — bilan")
print(f"  caractères : {total:,} -> {total_filtre:,} ({100 * (total - total_filtre) / total:.2f} % retirés)")
print(f"  {len(textes)} livres écrits dans {SORTIE.relative_to(RACINE)}/")
