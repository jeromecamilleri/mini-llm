# 01 — Extraire le texte d'un livre Kindle

Script : [`../scripts/01_extraction.py`](../scripts/01_extraction.py) · outil : [`../tools/mobi.py`](../tools/mobi.py)

## L'essentiel en 30 secondes

- Un `.mobi` / `.azw3` est un fichier binaire, pas du texte.
- Il contient le livre en **HTML**, découpé en blocs de 4096 octets, **compressés** (LZ77 « PalmDOC »).
- On lit les blocs, on les décompresse, on retire les balises HTML : il reste le texte brut.
- Testé sur 3 livres : ça marche, mais le texte n'est pas encore propre (leçon suivante).

## Ce qu'on a fait

| Livre | Format | Mots |
|---|---|---|
| Voltaire, *L'Ingénu* | `.azw3` | 25 963 |
| Erri De Luca, *Le poids du papillon* | `.mobi` | 13 255 |
| Bergounioux, *Faute d'égalité* | `.mobi` | 7 024 |

Les 353 fichiers ont la même compression et aucun DRM, donc la même méthode vaut pour tous.

## Les défauts à nettoyer

- L'Ingénu commence par l'en-tête légal du **Projet Gutenberg** (en anglais) : à retirer.
- Lettrine collée : « Àl’heure » au lieu de « À l’heure ».
- Titres, page de garde, éditeur en début de livre.

## Ce qu'on en déduit

- Décoder à la main est faisable : c'est un LZ77, quelques dizaines de lignes.
- Un livre fait 7 000 à 26 000 mots ; 353 livres donneront plusieurs millions de mots.
- Le texte extrait est écrit dans `datasets/extracted/` ; les originaux ne bougent pas.
