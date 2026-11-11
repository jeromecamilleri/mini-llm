"""06_corpus — Leçon 2 : nettoyer le texte extrait.

Le texte de la leçon 1 contient des déchets : en-tête légal anglais,
lettrines collées, guillemets typographiques de plusieurs sortes, lignes
coupées toutes les 70 caractères. Un modèle apprend ce qu'on lui montre :
si le corpus contient des déchets, il les apprend aussi.

On applique des règles simples (tools/nettoyage.py), une par une, et on
mesure ce qui change. Le résultat va dans datasets/cleaned/.
"""

import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nettoyage import (  # noqa: E402
    normaliser, nettoyer, refaire_paragraphes, reparer_lettrines, retirer_gutenberg,
)

EXTRAIT = RACINE / "datasets" / "extracted"
NETTOYE = RACINE / "datasets" / "cleaned"
NETTOYE.mkdir(parents=True, exist_ok=True)

fichiers = sorted(EXTRAIT.glob("*.txt"))


def inventaire(texte):
    """Caractères autres que lettres ASCII, chiffres et espace."""
    return "".join(sorted(c for c in set(texte) if not (c.isascii() and c.isalnum()) and c != " "))


print("ÉTAPE 1 — effet de chaque règle (nombre de caractères)")
for f in fichiers:
    t = f.read_text(encoding="utf-8")
    print(f"\n{f.stem[:40]}  brut : {len(t):,}")
    for nom, regle in [
        ("retirer_gutenberg", retirer_gutenberg),
        ("normaliser", normaliser),
        ("reparer_lettrines", reparer_lettrines),
        ("refaire_paragraphes", refaire_paragraphes),
    ]:
        avant, t = len(t), regle(t)
        print(f"  {nom:<20} {avant - len(t):>+9,}  -> {len(t):,}")

print("\nÉTAPE 2 — avant / après")
for f in fichiers:
    brut = f.read_text(encoding="utf-8")
    propre = nettoyer(brut)
    (NETTOYE / f.name).write_text(propre, encoding="utf-8")
    print(f"\n{f.stem[:40]}")
    print(f"  mots : {len(brut.split()):,} -> {len(propre.split()):,}")
    print(f"  paragraphes : {brut.count(chr(10)) + 1:,} -> {propre.count(chr(10)) + 1:,}")
    print(f"  symboles avant : {inventaire(brut)!r}")
    print(f"  symboles après : {inventaire(propre)!r}")
    print(f"  début : {propre[:200]!r}")

print("\nÉTAPE 3 — caractères les plus rares du corpus nettoyé")
compte = Counter("".join(f.read_text(encoding="utf-8") for f in NETTOYE.glob("*.txt")))
print(f"  {len(compte)} caractères différents")
print("  les plus rares :", [(c, n) for c, n in compte.most_common()[-12:]])
