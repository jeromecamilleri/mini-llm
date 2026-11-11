# Exercices 1 : le neurone

Cours associé : [`../../scripts/a_la_main/01_neurone.py`](../../scripts/a_la_main/01_neurone.py)

Un neurone calcule deux choses :

1. une **somme pondérée** : `z = x·w + b`
2. une **activation** (ici la sigmoïde) : `σ(z) = 1 / (1 + e^(−z))`

## Exemple résolu

Données : `x = [1, 2]`, `w = [3, -1]`, `b = 0.5`

**Étape 1 : somme pondérée**

```
z = 1·3 + 2·(−1) + 0.5
  = 3 − 2 + 0.5
  = 1.5
```

**Étape 2 : sigmoïde**

```
e^(−1.5) ≈ 0.2231
σ(1.5)   = 1 / (1 + 0.2231) ≈ 0.8176
```

> Piège classique : `1 / (1 + z)` est faux, il faut l'exponentielle `e^(−z)`.

**Étape 3 : conclusion**

- La sortie est toujours entre 0 et 1.
- `0.82 > 0.5` (équivalent à `z > 0`) : le neurone est **activé**.
- Il n'est pas saturé : il le serait vers 0.99, soit `z ≈ 4.6`.
- Si `z` avait été négatif (ex. `−1.5`), la sortie serait ≈ 0.18 : neurone plutôt inactif.

Le code correspondant est dans le script : étapes 3 et 5.

## Que fait-on de cette sortie ?

1. **Décider** : au-dessus de 0.5 → classe « 1 », sinon classe « 0 ».
2. **Comparer à la cible** : l'écart se mesure avec une fonction de perte.
3. **Apprendre** : on ajuste `w` et `b` pour réduire l'erreur (perceptron, puis rétropropagation).

## À retenir

- `σ(0) = 0.5` : à `z = 0`, le neurone est exactement indécis.
- Si tous les poids valent 0, `z = b` : la sortie ne dépend plus que du biais, pas de l'entrée.
