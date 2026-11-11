# Leçon 5 : les séquences d'entraînement

Script : [`05_sequences.py`](../scripts/05_sequences.py)
Lancement : `.venv/bin/python src/02_tokenizer/scripts/05_sequences.py`

## Le corpus utilisé

Fichier : [`corpus_animaux.txt`](../corpus_animaux.txt) (15 phrases, **67 tokens** en mots ; 10 mots différents, plus `.` et `'`).

> le chat dort. le chien dort. le cheval dort. l'oiseau dort. le chat mange. le chien mange. le cheval mange. l'oiseau mange. le chat court. le chien court. le cheval court. le chat dort aussi. le chien mange aussi. l'oiseau court aussi. le cheval dort aussi.

Tokenisé (leçon 2), le début donne : `le chat dort . le chien dort . …` soit les identifiants `[10, 4, 8, 2, 10, 6, 8, 2, …]`.

## Pourquoi cette leçon ?

**On cherche toujours le mot suivant.** La seule nouveauté : le modèle voit **plusieurs mots avant** de prédire, pas un seul.

| Leçon | Le modèle voit | Il doit prédire |
|---|---|---|
| 4 | `chat` (1 mot) | `dort` |
| 5 | `le chat dort .` (jusqu'à T mots) | `le` |

Pourquoi plusieurs mots ? Avec un seul mot, après `.` on ne sait pas quoi prédire. Avec `le chat dort .`, le modèle a plus d'indices.

Cette leçon ne fait que **préparer les données** : découper le texte (une longue suite d'identifiants) en exemples « voici les T derniers tokens → voici la suite ». L'entraînement viendra dans `03_transformer` puis `04_training`.

| Terme | Sens |
|---|---|
| **Fenêtre de contexte (T)** | nombre de tokens vus d'un coup (ici 4 ; un vrai GPT : 256 à plusieurs milliers) |
| **Entrée x** | T tokens consécutifs : `ids[i : i+T]` |
| **Cible y** | les mêmes tokens **décalés d'un cran** : `ids[i+1 : i+T+1]` |
| **Batch** | petit paquet de fenêtres traitées ensemble (leçon 10) |

## Le décalage d'un cran, pas à pas

Le décalage sert à fabriquer les réponses : chaque mot de `y` est le mot qui **suit** celui de `x` juste au-dessus.

| | 1er mot | 2e | 3e | 4e |
|---|---|---|---|---|
| x (entrée) | le | chat | dort | . |
| y (réponse) | chat | dort | . | le |

Une fenêtre contient donc **T exercices** de prédiction :

| Le modèle voit | Il doit prédire |
|---|---|
| le | chat |
| le chat | dort |
| le chat dort | . |
| le chat dort . | le |

Le modèle ne voit **jamais la suite** : pour prédire `dort`, il n'a que `le chat`.

## Chiffres

| Quantité | Valeur |
|---|---|
| Fenêtres possibles (glissement d'un token) | 67 − 4 = **63** |
| Exercices de prédiction | 63 × 4 = **252** |
| Entraînement / validation | 60 / 7 tokens (coupure à 90 %) |
| Forme d'un batch de 3 fenêtres | x et y : **(3, 4)** = (B, T) |

## Ce qu'on en déduit

- **Un seul texte, beaucoup d'exemples** : 67 tokens donnent 252 exercices, car les fenêtres se chevauchent.
- **Le modèle ne doit jamais voir la suite** : pour prédire « dort », il n'a que « le chat ». Un **masque** l'imposera dans l'attention (chapitre `03_transformer`, pas encore écrit).
- **Couper le texte avant de fabriquer les fenêtres** : sinon une fenêtre d'entraînement et une de validation se chevauchent, et la validation devient trop optimiste (leçon 11).
- **La méthode ne dépend pas du tokenizer** : mots, BPE ou caractères. Mais T tokens ≠ T mots : avec des caractères, une fenêtre couvre beaucoup moins de texte.
- **Le tenseur (B, T)** est exactement l'entrée du chapitre `03_transformer` : l'embedding la transforme en (B, T, dimension).

## Fin du chapitre `02_tokenizer`

On sait maintenant : tokeniser (caractères, mots, BPE), représenter chaque token par un vecteur appris, et fabriquer des batchs `(x, y)`.
Prochain chapitre : `03_transformer`, l'attention.
