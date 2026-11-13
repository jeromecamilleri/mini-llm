# 04 — Le réseau CNN : politique et valeur

Script : [`../scripts/04_reseau.py`](../scripts/04_reseau.py)
Outil : [`reseau.py`](../tools/reseau.py)

## L'essentiel en 30 secondes

- Le réseau lit la grille (10 canaux × 9 × 9) et sort **729 scores**, un par action : c'est la **politique**. Il sort aussi **un nombre** (la **valeur**), utile pour apprendre plus tard.
- Il est fait de **convolutions 3 × 3** : un petit filtre glissé sur la grille, mêmes poids partout.
- Les actions interdites (case remplie, doublon) sont **masquées** : probabilité exactement 0.
- Taille : **301 898 paramètres**, soit 60 fois moins que le GPT (17,8 M).
- Au départ, la dernière couche de la politique est à zéro : le réseau joue **exactement au hasard parmi les actions autorisées**. Mesuré : retour +23,7 et 90 coups (leçon 3 : +25 et 87).

## Les mots à connaître

| Mot | Sens |
|---|---|
| convolution | filtre 3 × 3 glissé sur la grille ; mêmes poids à chaque position |
| canal | une carte 9 × 9 de valeurs (10 en entrée, 64 dans les couches cachées) |
| politique | un score par action ; softmax → probabilités ; elle choisit le coup |
| valeur | un nombre par grille : ce qu'on espère encore gagner |
| masque | actions interdites mises à −10⁹ avant le softmax (probabilité 0) |
| résiduel | sortie = entrée + correction ; aide les réseaux profonds à apprendre |

## L'architecture

```
entrée (10, 9, 9)
  → conv 3×3 → 64 canaux
  → 4 blocs résiduels (2 convolutions 3×3 chacun)       8 convolutions : le champ de vue
  ├→ conv 1×1 → 9 canaux (un par chiffre) → 729 scores     couvre toute la grille
  └→ moyenne sur la grille → couche linéaire → valeur
```

Un filtre 3 × 3 ne voit que ses voisins. Chaque convolution élargit le champ de vue de 2 cases ; avec 9 convolutions, une case « voit » toute la grille, donc sa ligne, sa colonne et son bloc.

La sortie `(9 chiffres, 9, 9)` est réordonnée en `(ligne, colonne, chiffre)` puis aplatie : l'indice vaut `ligne × 81 + colonne × 9 + (chiffre − 1)`, comme `encode_action`.

## Pourquoi la dernière couche est à zéro

Si tous les scores sont égaux, le softmax masqué donne la même probabilité à chaque action autorisée : c'est le **hasard + masque** de la leçon 3. On part donc d'un repère connu (environ 90 coups) et on mesure tout gain par rapport à lui. Vérifié : probabilité maximale 0,0054 = 1 / 167 actions autorisées.

## Ce qui a été vérifié

| Contrôle | Résultat |
|---|---|
| encodage vectorisé = encodage de l'environnement | identique |
| masques vectorisés = masques de la leçon 3 (256 grilles) | 0 écart |
| probabilité des actions interdites | 0 |
| réseau non entraîné, 50 grilles faciles | 100 % résolues, +23,7, 90 coups |

## La vitesse (réseau seul, avant entraînement)

| Appareil | Lot | Décisions par seconde |
|---|---|---|
| CPU | 256 | 8 000 |
| GPU (GTX 1070) | 64 | 25 700 |
| GPU | 256 | 35 200 |
| GPU | 1 024 | 39 300 |

L'encodage d'un lot de 256 grilles passe de 7 ms (boucles de l'environnement) à **0,2 ms** (vectorisé).

Estimation, **non mesurée** : l'entraînement ajoute le calcul des gradients (environ 3 fois plus lent que la lecture seule) et le jeu des environnements (10 µs par coup, un seul cœur). On peut espérer environ 10 000 décisions par seconde, soit 30 à 40 millions par heure. Avec des parties d'environ 90 coups au départ, cela fait environ 400 000 parties par heure. La vraie vitesse sera mesurée à la leçon 06.

## Ce qu'on en déduit

- Le réseau est petit : on ne manquera ni de mémoire GPU ni de temps pour l'architecture elle-même. Le temps d'entraînement dépendra du **nombre de parties** nécessaires, pas de la taille du réseau.
- La politique, la valeur et le masque sont trois choses distinctes : le réseau apprend la politique ; le masque vient des règles ; la valeur sert à réduire le bruit de l'apprentissage.
- Partir d'une politique uniforme donne un point de départ mesuré, plus honnête qu'un réseau aléatoire dont on ignore le niveau.
