# 12 — Tokeniser le texte filtré

Script : [`../scripts/12_tokenisation_filtree.py`](../scripts/12_tokenisation_filtree.py)

## L'essentiel en 30 secondes

- On tokenise les livres filtrés (leçon 11) en `train_v2.bin`, avec **le même BPE** qu'avant.
- La validation n'est pas touchée : mêmes livres, **non filtrés**, même `validation.bin`.
- Résultat : 64 301 023 tokens contre 66 216 164 (2,89 % de moins), aucun token inconnu.

## Pourquoi tout garder identique sauf le texte d'entraînement

| Élément | Expérience D | Pourquoi |
|---|---|---|
| BPE | inchangé | mêmes ids, modèles comparables |
| livres de validation | inchangés | la perte mesure le même texte |
| `validation.bin` | non réécrit | idem |
| `train` | `train_v2.bin` | la seule variable |
| architecture, LR, étapes | ceux de la base choisie | une variable à la fois |

Dans la leçon 7, on choisit les données par `DONNEES_TRAIN = "train_v2"`.

## Ce qu'on en déduit

- Les 3 % de tokens en moins viennent surtout du code web : le BPE avait dépensé des fusions pour du CSS, qui sont maintenant rarement utilisées. C'est une petite perte de vocabulaire, qu'on ne corrige pas ici pour ne pas changer deux choses.
- Si D améliore la validation, ce sera uniquement grâce au texte plus propre.
