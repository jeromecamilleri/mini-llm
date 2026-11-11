# Leçon 3 : comparer des modèles sur les mêmes prompts

Script : [`03_comparaison.py`](../scripts/03_comparaison.py)
Lancement : `.venv/bin/python src/05_inference/scripts/03_comparaison.py`

## L'essentiel en 30 secondes

On compare **trois modèles qui ne diffèrent que d'une chose** : le moment de l'entraînement (règle du projet : une seule variable à la fois).

| Modèle | Ce que c'est |
|---|---|
| **au hasard** | jamais entraîné (poids tirés au hasard) |
| **meilleur** | checkpoint de l'étape 80 (validation minimale) |
| **dernier** | checkpoint de l'étape 600 (a appris par cœur) |

Même prompt, même réglage du tirage (`T = 1`), même graine : seul le modèle change.

Deux mesures :

| Mesure | Question |
|---|---|
| perte de validation | à quel point le modèle est surpris par du texte **jamais vu** |
| phrases correctes | quelle part des phrases générées suit la grammaire du corpus : `(le chat \| le chien \| le cheval \| l ' oiseau) (dort \| mange \| court) [aussi]` |

La grammaire ne sert qu'à **noter** le texte ; le modèle ne la connaît pas.

## Ce qu'on a lancé

Prompt `l ' oiseau`, tirage :

| Modèle | Texte |
|---|---|
| au hasard | `l ' oiseau chien l court . chien l ' . mange cheval ' '` |
| meilleur | `l ' oiseau mange . le chat chien mange . le cheval court . le` |
| dernier | `l ' oiseau mange . le chat court . le chien court . le cheval` |

200 générations de 24 mots (départ `le`) :

| Modèle | Perte validation | Phrases correctes |
|---|---|---|
| au hasard | 2,819 | **0 %** (0 / 399) |
| meilleur (étape 80) | **0,960** | 90,6 % (950 / 1048) |
| dernier (étape 600) | 2,353 | **98,6 %** (984 / 998) |

## Ce qu'on en déduit

- Le modèle **au hasard** n'écrit aucune phrase correcte : l'entraînement a bien tout changé, sans toucher à l'architecture.
- **Surprise** : le dernier modèle a la **pire** validation des deux entraînés, mais écrit le **meilleur** texte. Ce n'est pas un bug.
- Les deux mesures ne répondent pas à la même question. La validation mesure la surprise sur du texte **neuf** (7 mots ici : très bruitée). La grammaire notée ici est celle du corpus d'**entraînement**, que le dernier modèle a mémorisé : il la recrache avec peu de fautes.
- Mesurer, c'est aussi **découvrir que sa mesure ne dit pas ce qu'on croyait**. Il faut savoir ce que chaque mesure voit.
- Avec un vrai corpus (`datasets/`), validation et qualité du texte évolueront bien plus ensemble, et la grammaire ne pourra plus s'écrire : on lira les textes, on regardera la perplexité.
- C'est exactement le plan de `docs/Readme.md` : comparer des modèles qui ne diffèrent que par le **corpus**, avec les mêmes prompts.

## Et ensuite ?

Un **vrai corpus** (`datasets/`) pour les expériences A à D du plan global.
