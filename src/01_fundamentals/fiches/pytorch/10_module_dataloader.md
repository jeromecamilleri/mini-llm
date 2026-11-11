# Exercices : tenseurs, `nn.Module` et `DataLoader`

Cours associé : [`../../scripts/pytorch/10_module_dataloader.py`](../../scripts/pytorch/10_module_dataloader.py)

Même XOR, même réseau `[2, 4, 1]` (`tanh` puis `sigmoid`). La fiche 09 a montré que PyTorch calcule les gradients ; ici on range le réseau comme dans tout projet PyTorch, et c'est la structure qu'aura notre mini-GPT.

## Pourquoi cette leçon ?

**Le réseau ne change pas.** Mêmes 17 nombres, mêmes calculs, même résultat. Ce qui change, c'est la façon de les organiser dans le code.

Fiche 09, le réseau était **en vrac** :

```python
W1, b1, W2, b2 = ...                      # 4 variables à trimballer partout
perte = f(X, W1, b1, W2, b2)              # à passer à chaque appel
optimiseur = Adam([W1, b1, W2, b2])       # à lister à la main
```

Avec un GPT, il y aura des dizaines de couches et des millions de nombres : impossible de les suivre à la main. Fiche 10, le réseau est **dans une boîte** :

```python
reseau = ReseauXor()                      # une seule variable
sortie = reseau(X)                        # la propagation avant
optimiseur = Adam(reseau.parameters())    # PyTorch retrouve les 17 nombres seul
```

Trois outils, trois problèmes résolus :

| outil | problème | ce qu'il apporte |
|-------|----------|------------------|
| **formes des tenseurs** | un tableau ne s'emboîte pas avec un autre | on vérifie les dimensions à chaque étape |
| **`nn.Module`** | trop de variables en vrac | une boîte « réseau » qui contient les nombres et le calcul |
| **`DataLoader`** | trop de données pour tout traiter d'un coup | il distribue les exemples par paquets |

Pour le XOR (4 exemples, 17 nombres), aucun n'est nécessaire. On prend l'habitude ici, sur un cas simple, pour ne pas la découvrir en même temps que le Transformer.

## 1. Les formes des tenseurs

Règle : `(N, entrées) @ (entrées, sorties) → (N, sorties)`, avec N le nombre d'exemples.

| tenseur | forme |
|---------|-------|
| `X` | (4, 2) |
| `W1` | (2, 4) |
| `Z1 = X @ W1 + b1` | (4, 4) |
| `A1 = tanh(Z1)` | (4, 4) |
| `W2` | (4, 1) |
| `Z2 = A1 @ W2 + b2` | (4, 1) |

`b1` a la forme (4,) : il est ajouté à chacune des 4 lignes de `X @ W1` (c'est le *broadcast*). Les nombres sont en `float32` (4 octets), le format normal des réseaux.

## 2. Une classe `nn.Module`

```python
class ReseauXor(nn.Module):
    def __init__(...):  # déclare W1, b1, W2, b2 avec nn.Parameter
    def forward(self, x):  # la propagation avant de la fiche 03
```

`nn.Parameter` dit à PyTorch « ce tenseur est un réglage du réseau, apprends-le » : il est alors retrouvé automatiquement par `reseau.parameters()`, la liste que reçoit l'optimiseur. `forward` contient le calcul de la fiche 03 ; on l'appelle en écrivant `reseau(x)`. On n'utilise pas `nn.Linear` (une couche toute faite) : on garde la main sur `W` et `b`.

| paramètre | forme | nombres |
|-----------|-------|---------|
| `W1` | (2, 4) | 8 |
| `b1` | (4,) | 4 |
| `W2` | (4, 1) | 4 |
| `b2` | (1,) | 1 |
| **total** | | **17** |

Ce sont les 17 nombres de la fiche 06.

## 3. `Dataset` et `DataLoader`

Imagine un GPT à entraîner sur 200 livres : impossible de les présenter d'un coup au réseau (la mémoire de la GTX 1070 est limitée). On les présente **par paquets** : les *batchs*. Le `DataLoader` s'en charge : il découpe les données en paquets et les **mélange** à chaque époque, pour que le réseau ne voie pas toujours les exemples dans le même ordre.

Sur le XOR, avec des batchs de 2, une époque se déroule ainsi :

| batch | entrées | cibles |
|-------|---------|--------|
| 0 | (0,1) et (0,0) | 1 et 0 |
| 1 | (1,1) et (1,0) | 0 et 1 |

On voit les 4 exemples du XOR, rangés en 2 paquets de 2. À l'époque suivante, l'ordre sera différent. **Après chaque batch**, les poids sont corrigés : avec 2 batchs, il y a 2 corrections par époque (au lieu d'une seule quand on présente tout d'un coup).

Avec 4 exemples c'est inutile, mais c'est exactement la boucle qu'on utilisera pour le texte.

## 4. La boucle d'entraînement

Pour chaque batch : `zero_grad` → perte → `backward` → `step`. Une **époque** = tous les batchs vus une fois.

Adam 0,05, 10 graines, arrêt sous 0,01 (500 époques maximum) :

| taille du batch | réussites | époques (médiane) | pire cas |
|-----------------|-----------|-------------------|----------|
| 4 (tout le jeu) | 9/10 | 49 | 89 |
| 2 | 8/10 | 88 | 144 |
| 1 | 8/10 | 80 | 120 |

Réseau entraîné (graine 1, batch 4, 60 époques) :

| x1 | x2 | sortie |
|----|----|--------|
| 0 | 0 | 0.027 |
| 0 | 1 | 0.918 |
| 1 | 0 | 0.887 |
| 1 | 1 | 0.133 |

## Ce qu'on en déduit

- **Même mécanique, mieux rangée** : `nn.Module` regroupe les paramètres et la propagation avant, `DataLoader` gère les paquets, la boucle reste celle de la fiche 09. Un GPT aura la même forme, avec plus de couches.
- **Les formes sont la première chose à vérifier** : une grande partie des erreurs PyTorch sont des formes incompatibles, d'où l'intérêt de les afficher.
- **Plus petit batch, plus de corrections par époque** : le batch 1 donne 4 corrections par époque, mais chaque correction est plus bruitée, et il n'est pas plus rapide en époques que le batch 4. Pour un GPT, la taille du batch est surtout limitée par la mémoire de la GTX 1070.
- **Des échecs apparaissent** : une graine sur dix ne converge pas en 500 époques (sorties bloquées à 0,5). Le tirage aléatoire des poids de départ compte, comme à la fiche 07 ; PyTorch ne tire pas les mêmes nombres que NumPy, d'où ces résultats un peu différents.
- **CPU ici** : le XOR est trop petit pour la GPU (fiche 09). Pour un gros réseau, il suffirait de passer `appareil` à `"cuda"`.
- **Il reste dans la partie PyTorch** : la séparation entraînement / validation.
