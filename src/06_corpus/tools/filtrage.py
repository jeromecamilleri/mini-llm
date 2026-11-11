"""Filtrage du bruit restant dans les livres nettoyés (expérience D).

Le nettoyage de la leçon 2 règle la typographie. Il laisse passer des déchets
d'édition numérique : morceaux de feuilles de style (« @font-face { } »),
mentions d'éditeur, ISBN, adresses web, codes de mise en page, lignes sans
lettres, alphabets étrangers rares. Chaque règle travaille sur UNE ligne
(= un paragraphe) et dit si on la supprime.
"""

import re
from collections import Counter

# code de feuille de style ou de page web
RE_CODE_WEB = re.compile(r"[{}]|@font-face|</?\w+[^>]*>|&\w+;")
# mentions d'édition : ISBN, droits, adresses, noms de formats
RE_EDITION = re.compile(
    r"https?://|www\.|ISBN|©|copyright|table of contents|\.epub|\.mobi|1001ebooks",
    re.IGNORECASE,
)
# codes de mise en page isolés (« FL4103 », « U000 »)
RE_CODE_ISOLE = re.compile(r"[A-Z]{1,3}\d{3,}|U\d{3}")
# début de réplique ou de citation : une ligne courte y est normale
RE_DIALOGUE = re.compile(r"^\s*[-«—*]")

REGLES = ["code web", "édition", "code isolé", "peu de lettres"]


def regle_supprimant(ligne):
    """Nom de la règle qui supprime la ligne, ou None si on la garde."""
    if RE_CODE_WEB.search(ligne):
        return "code web"
    if RE_EDITION.search(ligne):
        return "édition"
    if RE_CODE_ISOLE.fullmatch(ligne.strip()) or "Dynamic layout" in ligne:
        return "code isolé"
    if len(ligne) >= 5 and not RE_DIALOGUE.match(ligne):
        if sum(c.isalpha() for c in ligne) / len(ligne) < 0.5:
            return "peu de lettres"
    return None


def alphabet_frequent(textes, seuil):
    """Caractères qui apparaissent au moins `seuil` fois dans l'ensemble du corpus."""
    compte = Counter()
    for t in textes:
        compte.update(t)
    return {c for c, n in compte.items() if n >= seuil} | {"\n"}


def filtrer(texte, alphabet, compteurs=None):
    """Retourne le texte sans les lignes de bruit.

    Une ligne dont plus de 10 % des caractères sont hors de l'alphabet (grec,
    coréen, symboles...) est supprimée ; sinon on retire juste ces caractères.
    `compteurs` (un Counter) reçoit le nombre de lignes retirées par règle.
    """
    gardees = []
    for ligne in texte.split("\n"):
        if not ligne:
            continue
        regle = regle_supprimant(ligne)
        if regle is None:
            hors = sum(c not in alphabet for c in ligne)
            if hors > 0.1 * len(ligne):
                regle = "alphabet rare"
            elif hors:
                ligne = "".join(c for c in ligne if c in alphabet)
                if compteurs is not None:
                    compteurs["caractères rares retirés (lignes)"] += 1
        if regle is None:
            gardees.append(ligne)
        elif compteurs is not None:
            compteurs[regle] += 1
    return "\n".join(gardees)
