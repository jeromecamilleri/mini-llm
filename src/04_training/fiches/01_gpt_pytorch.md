# Leçon 1 : le GPT en PyTorch et sa première perte

Script : [`01_gpt_pytorch.py`](../scripts/01_gpt_pytorch.py)
Code : [`tools/gpt_torch.py`](../tools/gpt_torch.py), [`tools/donnees.py`](../tools/donnees.py)
Lancement : `.venv/bin/python src/04_training/scripts/01_gpt_pytorch.py`

## L'essentiel en 30 secondes

Le GPT du chapitre `03_transformer` était en NumPy. NumPy **ne calcule pas les gradients**, or **entraîner = corriger les poids selon le gradient**. On réécrit donc la **même architecture** en PyTorch, qui calcule les gradients tout seul (`backward()`).

| | `03_transformer` (NumPy) | `04_training` (PyTorch) |
|---|---|---|
| architecture | embeddings + positions, blocs, couche finale | **la même** |
| gradients | non | **oui** (autograd) |
| données | une phrase `(T,)` | un batch `(B, T)` |

**B** = taille du batch (nombre de fenêtres traitées ensemble). **T** = longueur d'une fenêtre.

## Pourquoi passer à PyTorch dès maintenant ?

| Question | Réponse |
|---|---|
| Pourquoi pas NumPy ? | Entraîner demande le gradient de **tous** les paramètres (têtes, feed forward, LayerNorm, embeddings). En NumPy il faudrait écrire la rétropropagation à la main pour chaque couche (softmax, masque, LayerNorm, GELU…) : long et facile à rater. |
| Est-ce de la triche ? | Non. En `01_fundamentals`, tu as fait la rétropropagation à la main sur de petits réseaux : tu en connais le principe. `backward()` fait exactement la même chose automatiquement. |
| Qu'est-ce qui reste à la main ? | L'architecture et les mécanismes (attention, masque, softmax). On n'utilise que des briques de base : `Linear`, `Embedding`, `LayerNorm`, autograd. Jamais un GPT préfabriqué. |
| Comment être sûr que c'est le même modèle ? | L'étape 2 du script copie les poids du GPT NumPy dans le GPT PyTorch et compare les scores. |

## Vérification : PyTorch contre NumPy

On tire des poids au hasard pour la version NumPy (`d = 8`, 2 têtes, 2 blocs), on les **copie** dans le modèle PyTorch (`depuis_numpy`), puis on passe `le chat dort .` dans les deux.

| Mesure | Résultat |
|---|---|
| scores NumPy | `(4, 13)` |
| scores PyTorch | `(4, 13)` |
| écart maximal | 9·10⁻⁵ |

Un écart de l'ordre de 10⁻⁴ est normal : PyTorch calcule en `float32`, NumPy en `float64`, et l'epsilon de la LayerNorm est écrit un peu différemment. Les scores sont les mêmes : **seule la façon de l'écrire change**.

Détails de la copie : une couche `Linear` de PyTorch stocke sa matrice **transposée** (sortie, entrée), d'où les `.T` ; les biais de l'attention n'existent pas en NumPy, on les met à zéro ; la GELU de PyTorch est réglée sur la variante `tanh`, celle de `03_transformer` et de GPT-2.

## La perte (cross entropy)

Pour chaque position, le modèle donne une probabilité à chaque mot. La perte vaut **−log(probabilité donnée au vrai mot suivant)**, moyennée sur le batch.

| Probabilité du vrai mot | Perte |
|---|---|
| 100 % | 0 |
| 50 % | 0,69 |
| 1/13 (hésite entre 13 mots) | ln 13 = 2,56 |

## Ce qu'on a lancé

Vocabulaire V = 13, d = 32, 2 têtes, 2 blocs, T = 4 : **26 432 paramètres**.

| Mesure | Résultat |
|---|---|
| forme entrée → sortie | `(8, 4)` → `(8, 4, 13)` |
| perte avant apprentissage | 2,424 (ln 13 = 2,565) |
| gradient de `sortie.weight` | existe, forme `(13, 32)` |
| un pas de 0,1 dans le sens du gradient | perte 2,424 → 2,206 |
| changer le dernier mot | les scores du passé ne bougent pas (masque causal) |

## Ce qu'on en déduit

- La **même architecture** passe de NumPy à PyTorch : les scores sont identiques aux arrondis près. Seuls les gradients automatiques sont nouveaux.
- Un modèle qui ne sait rien a une perte proche de **ln(V)** : c'est le **point de départ** à battre.
- `loss.backward()` donne **un gradient par paramètre**. Un petit pas à l'opposé du gradient fait baisser la perte.
- **Entraîner = répéter ce pas des milliers de fois.**

## Et ensuite ?

Leçon 2 : la boucle d'entraînement complète (AdamW, validation, génération).
