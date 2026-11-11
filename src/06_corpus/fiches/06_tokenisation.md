# 06 — Tokeniser le corpus et garder des livres pour la validation

Script : [`../scripts/06_tokenisation.py`](../scripts/06_tokenisation.py)

## L'essentiel en 30 secondes

- Le modèle lit des **numéros** (ids de tokens), pas du texte : on convertit chaque livre avec le BPE de la leçon 5.
- On obtient deux fichiers : `train.bin` (entraînement) et `validation.bin` (livres **entiers** mis de côté, jamais vus).
- Chaque id est écrit sur 2 octets : 70 millions de tokens tiennent dans environ 140 Mo.

## Pourquoi des livres entiers pour la validation ?

| Méthode | Problème |
|---|---|
| morceaux pris au hasard dans tous les livres | le modèle a déjà vu les pages voisines (mêmes personnages, mêmes phrases) : la validation paraît meilleure qu'elle ne l'est |
| **livres entiers mis de côté** | mesure ce que le modèle sait faire sur un livre qu'il n'a jamais lu |

C'est l'idée de la leçon 2 de `04_training` : la perte d'entraînement baisse toujours, seule la validation dit si le modèle généralise ou apprend par cœur.

## Réglages

- `PART_VALIDATION = 0.05` : 5 % des livres, soit environ 17 sur 346.
- `GRAINE = 0` : le tirage est fixe, on retrouve toujours les mêmes livres (liste dans `datasets/tokenized/validation_livres.txt`). Changer la graine change la validation, donc les chiffres ne seraient plus comparables.

## À observer quand tu lances le script

- La liste des livres de validation : un livre d'un auteur peut y être alors que ses autres livres sont dans l'entraînement (c'est un peu plus facile pour le modèle que pour un auteur inconnu).
- Le nombre de tokens de chaque fichier, et le pourcentage de tokens inconnus `<inc>` : il doit être minuscule.
- Le début décodé : il redonne le texte du livre, ce qui prouve que rien n'est perdu.

## Ce qu'on en déduit

- Après cette leçon, le texte n'est plus utile pour l'entraînement : tout est dans `train.bin` et `validation.bin`.
- Un id est un simple numéro : le sens n'apparaîtra que dans les embeddings appris par le modèle.
- Les 611 caractères du tokenizer (dont beaucoup de très rares) pourront être réduits plus tard ; ce sera une amélioration à tester seule, car changer le tokenizer change les chiffres.
