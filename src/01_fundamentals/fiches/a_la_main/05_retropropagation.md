# Exercices : la rétropropagation

Cours associé : [`../../scripts/a_la_main/05_retropropagation.py`](../../scripts/a_la_main/05_retropropagation.py)

Même exemple que les fiches 03 et 04 : le XOR, réseau `[2, 4, 1]` (`tanh` puis `sigmoid`), perte MSE.

## Le problème

- Fiche 03 : on sait **calculer la sortie** (propagation avant).
- Fiche 04 : on sait **mesurer l'erreur** (MSE ≈ 0.2687).
- Il reste à savoir **comment modifier chacun des 17 paramètres** pour réduire cette erreur.

La réponse est le **gradient** de chaque paramètre : `dPerte/dPoids`.

| signe du gradient | signification | action |
|-------------------|---------------|--------|
| positif | augmenter ce poids **augmente** la perte | le baisser |
| négatif | augmenter ce poids **baisse** la perte | le monter |

## Comment ça marche : remonter le réseau

On part de la perte et on **remonte** vers l'entrée, couche par couche, avec la règle de dérivation en chaîne :
l'effet d'un poids sur la perte = effet sur `Z`, puis sur `A` (dérivée de l'activation, fiche 02), puis sur la perte.

```
entrée X -> [couche 1 : tanh] -> A1 -> [couche 2 : sigmoid] -> A2 -> perte
                                                                        |
  dW1, db1  <-  dZ1  <-  dA1  <-  dW2, db2  <-  dZ2  <-  dA2  <---------+
            (retour)
```

Formules, pour une couche :

```
dZ      = dA * activation'(Z)      # on traverse l'activation
dW      = A_prev.T @ dZ            # gradient des poids
db      = somme de dZ              # gradient des biais
dA_prev = dZ @ W.T                 # on transmet l'erreur à la couche d'avant
```

Chaque gradient a la **même forme** que le paramètre qu'il corrige : `dW1 (2, 4)`, `db1 (4,)`, `dW2 (4, 1)`, `db2 (1,)`.

## Exemple résolu

Réseau non entraîné sur les 4 cas du XOR :

| étape | résultat |
|-------|----------|
| sorties | 0.5, 0.3953, 0.4525, 0.3992 (MSE = 0.2687) |
| `dW2` | [0.0083, −0.0053, −0.0547, −0.0230] |
| `dPerte/dW1[0,0]` par rétropropagation | 0.01299302 |
| `dPerte/dW1[0,0]` par mesure directe | 0.01299302 |
| un pas de correction (taux 0.5) | perte 0.2687 → **0.2658** |

La **mesure directe** consiste à bouger un poids de `±ε` (10⁻⁵) et à regarder de combien la perte change. Elle donne la même valeur que la rétropropagation : le calcul est juste.

## Ce qu'on en déduit

- **Pourquoi pas la mesure directe pour tout ?** Elle demande de recalculer le réseau 2 fois **par paramètre**. Avec 17 paramètres, c'est 34 passages. Un GPT en a des milliards. La rétropropagation donne **tous** les gradients en un seul passage arrière : c'est ce qui rend l'entraînement possible.
- **C'est pour ça qu'on a besoin des dérivées** : `sigmoid_grad` et `tanh_grad` (fiche 02) servent ici. Une activation sans dérivée utilisable ne peut pas être entraînée ainsi.
- **Gradient qui disparaît** : à chaque couche traversée, le gradient est multiplié par `activation'(Z)`. Avec `sigmoid` (pente ≤ 0.25), il rétrécit vite dans un réseau profond, d'où l'intérêt de relu et GELU pour les couches cachées.
- **Un seul pas ne suffit pas** : la perte passe de 0.2687 à 0.2658, un progrès minuscule. Il faut répéter ce pas des centaines de fois : c'est la **descente de gradient** (prochaine fiche).
- **Lien avec le GPT** : un transformer s'entraîne exactement ainsi (perte, gradients par rétropropagation, correction des poids). Seule la perte change (entropie croisée au lieu de la MSE) et le réseau est bien plus grand. PyTorch calcule ces gradients automatiquement (`autograd`), mais c'est exactement ce calcul, que l'on aura écrit à la main d'abord.
