# Leçon 4 : les positions

Script : [`04_positions.py`](../scripts/04_positions.py)
Code : [`tools/attention.py`](../tools/attention.py) (`ajouter_positions`)
Lancement : `.venv/bin/python src/03_transformer/scripts/04_positions.py`

## Pourquoi cette leçon ?

L'attention (leçon 3) ne regarde que le **contenu** des vecteurs, jamais la **place** du token dans la phrase. Or l'ordre compte : « le chien mord l'homme » n'est pas « l'homme mord le chien ».

**Preuve sur nos phrases** : « le chat dort » et « chat le dort ». Les mots d'avant `dort` sont les mêmes, seulement permutés.

| Sans positions | Poids de `dort` sur ses prédécesseurs | Vecteur de `dort` |
|---|---|---|
| le chat dort | le 8 %, chat 77 % | `[1.65, 0.35]` |
| chat le dort | chat 77 %, le 8 % | `[1.65, 0.35]` |

Les poids suivent leurs mots, mais le résultat est **identique** : le mélange est une somme, et une somme ne dépend pas de l'ordre des termes. Le modèle ne peut pas distinguer les deux phrases.

## L'idée : ajouter la place au vecteur

Avant l'attention, on ajoute à chaque token un vecteur qui dépend de **sa position** :

`X[i] = E[i] + P[i]`

| Symbole | Sens |
|---|---|
| `E` | embedding du token (`02_tokenizer`, leçon 4), le même partout |
| `P` | **embedding de position** : un vecteur par place 0, 1, 2, … |
| `X` | ce qu'on donne à l'attention |

`P` est une table de paramètres **appris**, comme `E` (c'est la solution de GPT-2).

Exemple avec une table écrite à la main (place 0 : `[0, 0]`, place 1 : `[0.3, 0.1]`, place 2 : `[0.6, 0.2]`) :

| Token | E | + P | = X |
|---|---|---|---|
| chat à la place 0 | [2.0, 0.5] | [0, 0] | [2.0, 0.5] |
| chat à la place 1 | [2.0, 0.5] | [0.3, 0.1] | [2.3, 0.6] |

Le même mot `chat` a maintenant deux vecteurs selon sa place.

## Résultat avec positions

| Avec positions | Poids de `dort` | Vecteur de `dort` |
|---|---|---|
| le chat dort | le 2 %, chat 63 %, dort 34 % | `[1.85, 0.57]` |
| chat le dort | chat 49 %, le 6 %, dort 45 % | `[1.52, 0.62]` |

Les deux vecteurs de `dort` **diffèrent** : l'ordre est maintenant dans le calcul.

## Ce qu'on en déduit

- **L'attention seule ne voit pas l'ordre** : c'est une somme pondérée. Il faut le lui donner explicitement.
- **On ajoute, on ne remplace pas** : `E + P` garde le sens du mot et y ajoute sa place.
- **Une table a un nombre de lignes fixe** : GPT-2 a 1024 places × 768 dimensions = 786 432 nombres appris. On ne peut donc pas lire plus de 1024 tokens d'un coup : c'est la **fenêtre de contexte** (`02_tokenizer`, leçon 5).
- **Il existe d'autres méthodes** : des modèles récents font tourner Q et K selon la place (RoPE) au lieu d'ajouter un vecteur. Même but, mais on ne les programme pas ici.
- **Dans notre mini-GPT** : une table `nn.Embedding(T, d)`, comme celle des tokens, apprise avec le reste.

## Et ensuite ?

Leçon 5 : **plusieurs têtes**. Une seule attention pose une seule question à la fois ; plusieurs têtes en parallèle peuvent chercher des choses différentes (par exemple qui fait l'action, et quel est le mot précédent).
