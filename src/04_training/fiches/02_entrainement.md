# Leçon 2 : la boucle d'entraînement

Script : [`02_entrainement.py`](../scripts/02_entrainement.py)
Lancement : `.venv/bin/python src/04_training/scripts/02_entrainement.py`

## L'essentiel en 30 secondes

On répète des milliers de fois le petit pas de la leçon 1. La boucle est **la même qu'en `01_fundamentals`** :

```python
for etape in range(N):
    x, y = tirer_batch(...)     # des fenêtres au hasard
    l = perte(modele, x, y)     # à quel point on se trompe
    optimiseur.zero_grad()      # on efface les anciens gradients
    l.backward()                # nouveaux gradients
    optimiseur.step()           # AdamW corrige les poids
```

| Notion | Rappel |
|---|---|
| **AdamW** | Adam + un léger « fondu » des poids, qui limite le surapprentissage |
| **perte de validation** | mesurée sur des mots **jamais vus** (la fin du corpus) |
| **perplexité** | `exp(perte)` : parmi combien de mots hésite-t-on en moyenne ? |

La perplexité de départ vaut V (hésite entre tous les mots). Plus elle approche de 1, plus le modèle est sûr de lui.

## Ce qu'on a lancé

600 étapes, batch de 16 fenêtres, `lr = 0,003`, 26 432 paramètres.

| Étape | Perte train | Perte validation |
|---|---|---|
| 0 | 2,536 | 2,819 |
| 100 | 0,609 | **1,164** |
| 200 | 0,503 | 1,886 |
| 300 | 0,493 | 2,129 |
| 600 | 0,463 | 2,353 |

Génération **avant** : `le chat oiseau chat . chat . chat . chat . chat` (charabia).
Génération **après**, glouton : `le chat dort . le chien dort . le cheval dort .` (du vrai texte !).
Avec tirage, température 0,5 ou 1 : `le chat mange . le chien mange aussi . l ' oiseau`.

## Ce qu'on en déduit

- La perte d'entraînement passe de 2,5 à 0,46 : le modèle a **appris** la structure « sujet verbe . ».
- Elle ne descend pas à 0 car le corpus est **ambigu** : après `le chat`, `dort`, `mange` ou `court` sont tous corrects.
- **Surapprentissage** : la validation est la meilleure à l'étape 100 (1,16) puis **remonte** pendant que l'entraînement baisse. Le modèle mémorise ses 60 mots. C'est le signal de l'**arrêt anticipé** (`01_fundamentals`, leçon 11).
- La validation ne compte que 7 mots : la mesure est très **bruitée**. Un vrai corpus (`datasets/`) la rendra fiable.
- Le même modèle donne du charabia ou du texte correct : **seuls les poids ont changé**.

## Et ensuite ?

Leçon 3 : sauvegarder les poids (checkpoints), garder le **meilleur** (validation minimale) et suivre les métriques.
