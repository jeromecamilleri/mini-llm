"""06_corpus — Leçon 16 : nettoyer le théâtre importé (Claire, TheatreGratuit).

Le point de départ : `datasets/claire/train.txt`, téléchargé sur Hugging Face
(corpus Claire, partie « TheatreGratuit »). Une réplique par ligne, au format
`[PERSONNAGE:] texte`, une ligne vide entre deux scènes. Notre modèle, lui, a
appris sur des livres où une réplique s'écrit `- texte` (c'est aussi le format du
chat, `10_chat.py`). Il faut donc convertir, et retirer ce qui fausserait
l'apprentissage.

Les règles (chacune a été choisie après avoir regardé des exemples réels) :

  1. FORMAT. `[VILDAMOUR:] Ah !` devient `- Ah !`. Le nom du personnage est retiré :
     le modèle apprendrait sinon des noms propres de pièces (4 443 noms distincts).
  2. DIDASCALIES. Dans ce corpus, certaines indications de scène (« et MADAME
     CARBONEL sortent. ») se retrouvent collées dans une réplique, avec le nom du
     personnage en MAJUSCULES. On supprime toute réplique qui contient en majuscules
     un nom de personnage du corpus (les cris comme « NON ! » restent).
  3. TEXTE DÉFORMÉ. Quelques répliques imitent un accent ou un bégaiement
     (« Voui, répêns-toâ, comme moâ je mê souis répêntîe. »). Le modèle les prendrait
     pour du français. On supprime toute réplique dont au moins 30 % des mots
     (5 mots ou plus) n'existent pas dans nos 346 livres. Le vocabulaire est celui
     des livres (mots vus au moins 5 fois), sans dictionnaire.
  4. RÉPLIQUES VIDES ou d'un seul caractère : supprimées.
  5. SCÈNES. La ligne vide entre deux scènes est conservée (une seule à la fois).

Ce qu'on NE touche pas : les répliques longues (monologues), les rares phrases
en anglais, l'orthographe ancienne. On les compte, c'est tout.

Écriture dans `datasets/claire/theatre_propre.txt` (dossier ignoré par git ; le fichier
d'origine n'est jamais modifié). Le script compte aussi les tokens avant et après,
avec le BPE existant (on ne refait pas de tokenizer : une seule variable change).

Attention : `datasets/claire/` est sous licence CC BY-NC-SA : ne jamais le commiter
ni le republier tel quel.

Lancement : .venv/bin/python src/06_corpus/scripts/16_nettoyage_theatre.py
"""

import collections
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bpe_rapide import BPE  # noqa: E402

ENTREE = RACINE / "datasets" / "claire" / "train.txt"
SORTIE = RACINE / "datasets" / "claire" / "theatre_propre.txt"
LIVRES = RACINE / "datasets" / "cleaned_v2"
BPE_JSON = RACINE / "datasets" / "tokenized" / "bpe_4000.json"
SEUIL_INCONNUS = 0.30  # part de mots absents des livres au-delà de laquelle on supprime
MIN_MOTS = 5  # en dessous, la proportion n'est pas fiable
MIN_OCCURRENCES = 5  # un mot est « connu » s'il apparaît au moins 5 fois dans les livres

MOTS = re.compile(r"[^\W\d_]+")  # lettres seulement : « Qu'est-ce » donne « Qu », « est », « ce »
LIGNE = re.compile(r"^\[([^\]]*?):\] ?(.*)$")

print("ÉTAPE 1 — lire le fichier")
lignes = ENTREE.read_text(encoding="utf-8").split("\n")
repliques = []  # (personnage, texte) ou None pour une ligne vide
for ligne in lignes:
    if not ligne.strip():
        repliques.append(None)
        continue
    m = LIGNE.match(ligne)
    if m is None:
        raise ValueError(f"ligne hors format : {ligne[:80]!r}")
    repliques.append((m.group(1), m.group(2).strip()))
n_repliques = sum(1 for r in repliques if r is not None)
personnages = {p for r in repliques if r for p in [r[0]]}
print(f"  {n_repliques:,} répliques, {len(personnages):,} personnages distincts")

print("\nÉTAPE 2 — le vocabulaire des livres (sans dictionnaire)")
compteur = collections.Counter()
for f in sorted(LIVRES.glob("*.txt")):
    compteur.update(w.lower() for w in MOTS.findall(f.read_text(encoding="utf-8")))
vocabulaire = {w for w, n in compteur.items() if n >= MIN_OCCURRENCES}
print(f"  {len(vocabulaire):,} mots connus (vus au moins {MIN_OCCURRENCES} fois)")

# Mots de personnages en majuscules (« CARBONEL », « MADAME ») : on ne garde que ceux de 3 lettres ou plus
noms_majuscules = {m for p in personnages for m in re.findall(r"[A-ZÀ-Ý]{3,}", p)}


def est_didascalie(texte):
    return any(mot in noms_majuscules for mot in re.findall(r"[A-ZÀ-Ý]{3,}", texte))


def part_inconnue(texte):
    mots = [w.lower() for w in MOTS.findall(texte)]
    if len(mots) < MIN_MOTS:
        return 0.0
    return sum(1 for w in mots if w not in vocabulaire) / len(mots)


print("\nÉTAPE 3 — filtrer")
rejets = collections.Counter()
exemples = collections.defaultdict(list)
propre = []
for r in repliques:
    if r is None:
        if propre and propre[-1] != "":
            propre.append("")
        continue
    texte = r[1]
    if len(texte) <= 1:
        raison = "vide ou 1 caractère"
    elif est_didascalie(texte):
        raison = "didascalie (nom en majuscules)"
    elif part_inconnue(texte) >= SEUIL_INCONNUS:
        raison = "texte déformé (mots inconnus)"
    else:
        propre.append(f"- {texte}")
        continue
    rejets[raison] += 1
    if len(exemples[raison]) < 4:
        exemples[raison].append(texte[:100])

for raison, n in rejets.most_common():
    print(f"  supprimées — {raison} : {n:,} ({n / n_repliques:.2%})")
    for e in exemples[raison]:
        print("      ", repr(e))
gardees = sum(1 for l in propre if l)
print(f"  gardées : {gardees:,} sur {n_repliques:,} ({gardees / n_repliques:.1%})")
longues = sum(1 for l in propre if len(l) > 1500)
print(f"  dont répliques de plus de 1 500 caractères (gardées) : {longues:,}")

print("\nÉTAPE 4 — écrire le fichier propre")
while propre and propre[-1] == "":
    propre.pop()
texte_propre = "\n".join(propre) + "\n"
SORTIE.write_text(texte_propre, encoding="utf-8")
print(f"  {SORTIE.relative_to(RACINE)} : {len(texte_propre):,} caractères")

print("\nÉTAPE 5 — compter les tokens (BPE existant)")
bpe = BPE.charger(BPE_JSON)
avant = len(bpe.encoder(ENTREE.read_text(encoding="utf-8")))
apres_ids = bpe.encoder(texte_propre)
inconnus = sum(1 for i in apres_ids if i == 0)
print(f"  avant : {avant:,} tokens")
print(f"  après : {len(apres_ids):,} tokens ({len(apres_ids) / avant:.0%} de l'original)")
print(f"  tokens inconnus : {inconnus:,} ({inconnus / len(apres_ids):.4%})")
print("  rappel : le corpus de livres d'entraînement fait 66 216 164 tokens")
print(f"  part du théâtre si on l'ajoute : {len(apres_ids) / (66_216_164 + len(apres_ids)):.1%}")

print("\nÉTAPE 6 — un extrait lisible")
print("\n".join(texte_propre.split("\n")[:12]))
