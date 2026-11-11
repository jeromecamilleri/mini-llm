# Exercices : la propagation avant d'un réseau

Cours associé : [`../../scripts/a_la_main/03_propagation_avant.py`](../../scripts/a_la_main/03_propagation_avant.py) (module : [`../../tools/forward.py`](../../tools/forward.py))

Chaque couche calcule `Z = A_prev @ W + b`, puis `A = activation(Z)`.

## Que veut dire `[2, 4, 1]` ?

C'est la **forme du réseau** : le nombre de neurones par couche, de l'entrée à la sortie.

```
 entrée      couche cachée      sortie
   2     ->        4        ->    1
 (x1, x2)    (4 neurones)    (1 neurone)
```

- **2** : deux nombres en entrée (pour le XOR, `x1` et `x2`, valant 0 ou 1).
- **4** : une couche cachée de 4 neurones. Chacun reçoit les 2 entrées et calcule sa propre somme pondérée puis son activation (`tanh`). Elle est « cachée » car on ne voit ni ses entrées ni ses sorties depuis l'extérieur.
- **1** : un neurone de sortie. Il reçoit les 4 sorties de la couche cachée et produit un nombre entre 0 et 1 (`sigmoid`) : la prédiction.

**Nombre de paramètres à apprendre**

| couche | poids | biais |
|--------|-------|-------|
| 1 (2 → 4) | 2 × 4 = 8 (matrice `W1`, forme (2, 4)) | 4 |
| 2 (4 → 1) | 4 × 1 = 4 (matrice `W2`, forme (4, 1)) | 1 |

Total : 8 + 4 + 4 + 1 = **17 paramètres**.

## Exemple résolu

Réseau `[2, 4, 1]` (activations `tanh` puis `sigmoid`), sur les 4 entrées du XOR, sans entraînement.

**Formes des tableaux** (4 exemples) :

```
Couche 1 : X (4, 2) @ W1 (2, 4) -> Z (4, 4)   # 4 exemples, 4 neurones cachés
Couche 2 : A1 (4, 4) @ W2 (4, 1) -> Z (4, 1)   # 4 exemples, 1 sortie
```

Attention : le premier « 4 » de `X` et de `Z` est le **nombre d'exemples** (les 4 cas du XOR), pas des neurones. Dans `Z (4, 4)`, ce sont donc 4 exemples × 4 neurones cachés.

**Sorties obtenues** : `[0.5, 0.3953, 0.4525, 0.3992]`, cibles XOR : `[0, 1, 1, 0]`.

**Perte** : `MSE ≈ 0.2687`.

### C'est quoi la MSE ?

**MSE** = *Mean Squared Error* (erreur quadratique moyenne). C'est la **perte** : un seul nombre qui mesure à quel point le réseau se trompe.

```
MSE = moyenne((sortie − cible)²)
```

| entrée | cible | sortie | erreur² |
|--------|-------|--------|---------|
| [0, 0] | 0 | 0.5000 | 0.2500 |
| [0, 1] | 1 | 0.3953 | 0.3657 |
| [1, 0] | 1 | 0.4525 | 0.2998 |
| [1, 1] | 0 | 0.3992 | 0.1594 |

Moyenne : (0.2500 + 0.3657 + 0.2998 + 0.1594) / 4 ≈ **0.2687**.

Comment lire ce nombre :
- **0** : le réseau est parfait.
- **0.25** : le réseau sort toujours 0.5, ce qui équivaut au hasard (erreur de 0.5, au carré, pour chaque cas).
- **Ici 0.27** : à peine pire que le hasard, le réseau n'a rien appris.

Pourquoi le carré ?
- Les erreurs sont toujours positives : −0.5 et +0.5 ne s'annulent pas.
- Les grosses erreurs sont davantage pénalisées.
- Il est facile à dériver, ce qui sert à la rétropropagation.

L'entraînement cherchera à faire descendre ce nombre vers 0.

## Conclusions

- Les sorties sont toutes proches de 0.5 : le réseau n'a rien appris, il ne distingue pas les cas.
- La MSE vaut environ 0.27. Une sortie constante 0.5 donnerait exactement 0.25, donc le réseau est au niveau du hasard.
- Pour le 1er exemple `[0, 0]`, `Z1 = 0 @ W1 + b = 0` car les biais sont initialisés à 0, donc `tanh(0) = 0` et la sortie vaut `sigmoid(0) = 0.5`. C'est le `0.5` de la première sortie.
- Les poids sont tirés avec un écart-type `sqrt(2 / n_entrées)` (initialisation de He) pour garder des activations d'amplitude raisonnable d'une couche à l'autre.
- Pour que le réseau apprenne le XOR (non séparable linéairement), il faudra la rétropropagation (labs suivants) : ici on ne fait que la propagation avant.
