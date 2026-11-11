# Exercices : PyTorch fait la rétropropagation

Cours associé : [`../../scripts/pytorch/09_pytorch.py`](../../scripts/pytorch/09_pytorch.py) (compare avec [`../../tools/backpropagation.py`](../../tools/backpropagation.py))

Même XOR, même réseau `[2, 4, 1]` (`tanh` puis `sigmoid`), même perte. Jusqu'ici on a écrit la rétropropagation à la main. Ici PyTorch la fait tout seul, et on vérifie qu'il trouve la même chose.

## Comment PyTorch calcule les gradients

Un tenseur créé avec `requires_grad=True` note chaque opération faite avec lui. `perte.backward()` rejoue ces opérations à l'envers (la règle de dérivation en chaîne de la fiche 05) et remplit `.grad` de chaque paramètre. Ce mécanisme s'appelle **autograd**.

| à la main (fiches 5 à 8) | PyTorch |
|---|---|
| `retropropagation(net, X, y)` | `perte.backward()` |
| `net.W[0] -= taux * dW1` | `optimiseur.step()` |
| classe `Adam` (fiche 8) | `torch.optim.Adam` |
| NumPy, CPU | tenseurs, CPU ou GPU |

## Résultat : mêmes poids de départ, mêmes gradients

Perte de départ : 0.2687 des deux côtés. Écart entre nos gradients et ceux de PyTorch :

| gradient | forme | écart maximal |
|----------|-------|---------------|
| `dW1` | (2, 4) | 1.4e-17 |
| `db1` | (4,) | 1.4e-17 |
| `dW2` | (4, 1) | 2.1e-17 |
| `db2` | (1,) | 2.8e-17 |

Un écart de 1e-17 n'est qu'une erreur d'arrondi : **nos 17 gradients faits main sont exacts.**

## Entraînement avec PyTorch

`torch.optim.Adam`, taux 0.05, graine 0 : perte sous 0.01 à l'époque 107 (notre Adam : ~106 pour cette graine, la plus lente des 10).

| x1 | x2 | sortie |
|----|----|--------|
| 0 | 0 | 0.051 |
| 0 | 1 | 0.878 |
| 1 | 0 | 0.943 |
| 1 | 1 | 0.130 |

Le XOR est appris. Le script montre aussi que `W1` peut être envoyé sur la GTX 1070 (`cuda:0`).

## CPU contre GPU

500 pas d'entraînement (perte, `backward`, Adam), mêmes calculs sur les deux appareils :

| réseau | CPU | GPU | le plus rapide |
|--------|-----|-----|----------------|
| XOR `[2, 4, 1]`, 4 exemples | 0,20 s | 0,35 s | **CPU**, 1,7 fois |
| large `[256, 2048, 1]`, 4096 exemples | 25,6 s | 1,9 s | **GPU**, 13,7 fois |

Même sur le petit XOR, la différence se voit, mais **dans le mauvais sens** : la GPU est plus lente. Chaque petite opération doit être lancée sur la carte (un coût fixe), et le calcul lui-même est trop petit pour le rentabiliser. Sur le gros réseau, des milliers de multiplications se font en parallèle et la GPU gagne largement. Un GPT est dans ce second cas.

(les temps varient d'une exécution à l'autre ; l'ordre de grandeur reste le même.)

## Ce qu'on en déduit

- **Notre compréhension est validée** : le calcul à la main (fiche 05) et autograd donnent le même résultat. Autograd n'est pas magique, c'est la même rétropropagation, automatisée.
- **On n'écrira plus les gradients à la main.** Pour un GPT avec des millions de paramètres, ce serait irréaliste et source d'erreurs. À partir de cette leçon, `backward()` s'en charge.
- **Ce qu'il reste à écrire soi-même** : l'architecture (couches, activations), la perte, la boucle d'entraînement. C'est là que le projet apprend quelque chose.
- **La GPU n'aide pas toujours** : sur le XOR elle est 1,7 fois plus lente que le CPU, sur un réseau 500 fois plus large elle est 13,7 fois plus rapide. Elle ne devient utile qu'à partir d'une certaine taille, et c'est pour cela qu'on la garde pour le GPT.
- **Float64 ici, float32 ensuite** : on a utilisé 64 bits pour comparer au chiffre près. Les vrais entraînements utilisent 32 bits ou moins, pour la vitesse et la mémoire de la GTX 1070.
- **Lien avec le GPT** : la boucle `zero_grad` → perte → `backward` → `step` est exactement celle qui entraîne les LLM.
