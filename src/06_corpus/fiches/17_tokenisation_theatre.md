# 17 — Tokeniser le théâtre et réserver des dialogues de validation

Script : [`../scripts/17_tokenisation_theatre.py`](../scripts/17_tokenisation_theatre.py)

## L'essentiel en 30 secondes

- Après la leçon 16 : 5,02 M de tokens de théâtre propre. On les transforme en fichiers `.bin` et on **met de côté 5,5 % pour mesurer**.
- Cette validation de **dialogues** montrera l'effet du théâtre. Sur nos 17 livres, il serait presque invisible.
- Le BPE ne change pas : une seule variable change, le corpus.
- Fichier d'entraînement du mélange simple : `train_theatre.bin`, **70 959 954 tokens** (66,2 M de livres + 4,74 M de théâtre, soit **6,7 %** de théâtre).

## Comment on coupe

Les pièces se suivent dans le fichier. On prend la **fin** (environ 5 %) et on coupe sur une **ligne vide** (changement de scène).

| Fichier | Tokens | Rôle |
|---|---|---|
| `theatre_train.bin` | 4 743 790 (94,5 %) | théâtre appris |
| `theatre_validation.bin` | 277 026 (5,5 %) | dialogues jamais appris |
| `train_theatre.bin` | 70 959 954 | livres + théâtre appris (option « mélange simple ») |

Pourquoi pas au hasard, réplique par réplique ? Les répliques voisines d'une même scène se ressemblent. Une réplique de validation tirée au milieu d'une scène apprise serait trop facile à prédire : la validation mentirait. En coupant sur une scène, le modèle n'a jamais vu ces personnages ni ces répliques.

## Pourquoi mettre les deux corpus bout à bout

L'entraînement tire ses fenêtres de 256 tokens **au hasard dans tout le tableau** (`tirer_batch`). L'ordre dans le fichier ne compte donc pas : chaque batch mélange livres et théâtre, en proportion de leur taille.

## Ce qu'on en déduit

- Le théâtre ne pèse que **6,7 %** du mélange : l'effet sera faible, mais on pourra le mesurer sur les dialogues.
- Pour savoir si le modèle dialogue mieux, il faudra mesurer **deux pertes** : celle des 17 livres (a-t-on perdu quelque chose ?) et celle des dialogues (a-t-on gagné ?).
- Cette validation est petite (277 026 tokens, quelques pièces). Elle donne un ordre de grandeur, pas une mesure fine.
