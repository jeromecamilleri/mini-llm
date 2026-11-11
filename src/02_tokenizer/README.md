# 02 — Représentation du langage

## Objectif global

Comprendre comment du **texte devient une suite de nombres** utilisable par un réseau de neurones, et comment ce choix influence la taille des données et du modèle.

Le fil rouge change : ce n'est plus le XOR, mais un **mini-texte écrit à la main** (quelques phrases sur des animaux), pour tout voir à l'œil nu.

## À la fin de ce chapitre, on sait

- encoder un texte en identifiants, et le décoder sans rien perdre ;
- comparer trois découpages : caractères, mots, sous-mots ;
- construire à la main un tokenizer **BPE**, comme celui des vrais GPT ;
- choisir la taille du vocabulaire en connaissance de cause ;
- donner un petit vecteur appris à chaque token (embeddings) ;
- préparer les séquences d'entraînement : fenêtres, cibles décalées, découpe entraînement/validation.

## Parcours

| leçon | thème | idée |
|-------|-------|------|
| 01 | caractères | petit vocabulaire, longues séquences |
| 02 | mots | séquences courtes, mais mots inconnus et vocabulaire qui explose |
| 03 | BPE (sous-mots) | le compromis : fusionner les paires les plus fréquentes |
| 04 | embeddings | un petit vecteur appris par token : les mots proches se rapprochent |
| 05 | séquences d'entraînement | fenêtres de T mots, cible décalée d'un cran, batchs `(B, T)` |

## Organisation

- `scripts/` : les leçons, des scripts Python à lire de haut en bas ;
- `fiches/` : une fiche `.md` par leçon (même nom), avec « Ce qu'on en déduit » ;
- `tools/` : `vocabulary.py`, `tokenizer_basic.py`, `bpe.py` ;
- `corpus_animaux.txt` : le mini-texte (15 phrases, 67 mots) partagé par les leçons 04 et 05, puis par les chapitres suivants.

Lancement, depuis la racine du dépôt :

```bash
.venv/bin/python src/02_tokenizer/scripts/01_caracteres.py
```

## Pourquoi ce chapitre

Le tokenizer est la porte d'entrée de tout le reste : le modèle ne voit jamais de texte, seulement des identifiants. Le choix du tokenizer (et donc du corpus sur lequel on le construit) change la taille du vocabulaire, la longueur des séquences et le coût de l'entraînement sur la GTX 1070.

## Pour illustrer : du texte aux vecteurs

Les blocs 2 à 4 de cette page (données, tokenisation, embeddings) correspondent à ce chapitre : le texte est découpé en tokens, chaque token devient un numéro, puis un vecteur (image externe, auteure Pooja Pawar, non reproduite par nous).

![Large Language Model, vue d'ensemble](../../img/Image%20ChatGPT%208%20nov.%202026,%2021_18_15.jpg)
