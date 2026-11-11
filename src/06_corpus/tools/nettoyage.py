"""Nettoyage du texte extrait d'un livre : des règles simples, une à une."""

import re

# Caractères typographiques ramenés à leur forme simple : moins de symboles
# différents pour dire la même chose = un vocabulaire plus petit à apprendre.
REMPLACEMENTS = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
    "\u00a0": " ", "\u202f": " ",
}


def retirer_gutenberg(texte):
    """Retire l'en-tête et le bloc légal du Projet Gutenberg (s'ils existent)."""
    if "Project Gutenberg" not in texte[:3000]:
        return texte
    lignes = texte.split("\n")
    debut = next((i for i, l in enumerate(lignes) if l.strip() == "***"), -1) + 1
    fins = [i for i, l in enumerate(lignes) if "This file should be named" in l]
    fin = fins[-1] if fins and fins[-1] > debut else len(lignes)
    return "\n".join(lignes[debut:fin])


def normaliser(texte):
    for ancien, nouveau in REMPLACEMENTS.items():
        texte = texte.replace(ancien, nouveau)
    texte = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", texte)  # caractères de contrôle
    return texte


def reparer_lettrines(texte):
    """« Àl'heure » au début d'un paragraphe -> « À l'heure » (lettrine collée)."""
    return re.sub(r"(?m)^À(?=[a-zéèêàâîôû])", "À ", texte)


def refaire_paragraphes(texte):
    """Une ligne = un paragraphe. Les retours à la ligne au milieu d'une phrase
    (lignes de 70 caractères des vieux fichiers) deviennent des espaces."""
    paragraphes = re.split(r"\n\s*\n", texte)
    paragraphes = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphes]
    paragraphes = [re.sub(r" {2,}", " ", p) for p in paragraphes]
    # une ligne qui ne contient qu'un numéro de page n'est pas du texte
    paragraphes = [p for p in paragraphes if p and not re.fullmatch(r"[\d\s.\-]+", p)]
    return "\n".join(paragraphes)


def nettoyer(texte):
    texte = retirer_gutenberg(texte)
    texte = normaliser(texte)
    texte = reparer_lettrines(texte)
    return refaire_paragraphes(texte)
