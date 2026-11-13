# 05 — L'environnement par lots

Script : [`../scripts/05_env_lots.py`](../scripts/05_env_lots.py)
Outil : [`env_lots.py`](../tools/env_lots.py)

## L'essentiel en 30 secondes

- `SudokuEnv` joue **une** grille à la fois. Le réseau, lui, veut décider pour **256 grilles** d'un coup. `SudokuEnvLots` joue B grilles en parallèle, dans des tenseurs PyTorch, sans boucle Python.
- Les règles et les récompenses sont **identiques** à `SudokuEnv`, qui reste la référence. Vérifié coup par coup : 0 écart sur 17 886 coups.
- Vitesse mesurée (lot de 256, coups au hasard parmi les autorisés) : 8 347 coups/s pour l'ancien montage, 49 726 sur CPU, **191 154 sur GPU**.
- L'environnement n'est plus le goulot : c'est le réseau (≈ 35 000 décisions/s) qui le devient.

## Les mots à connaître

| Mot | Sens |
|---|---|
| lot (batch) | B grilles jouées en même temps (ici 256) |
| présence | `ligne_a[l, d]` : le chiffre d+1 est déjà dans la ligne l (idem colonne, bloc) |
| timeout | partie coupée au bout de `n_coups_max` coups, sans être finie |
| gelée | une partie terminée ignore les actions suivantes et rapporte 0 |

## La structure de données

Par grille : la grille `(9, 9)`, la solution `(9, 9)`, et trois tableaux de booléens `(9, 9)` : présence par ligne, par colonne, par bloc (bloc = `3 × (ligne // 3) + colonne // 3`).

Poser un chiffre met à jour 3 booléens. Le masque des actions autorisées se déduit des trois tableaux (case vide et chiffre absent de la ligne, de la colonne et du bloc), sans relire toute la grille. Gain mesuré : voir ci-dessus ; la part due à la structure seule, sans le GPU, n'est pas isolée.

## Le timeout : c'est quoi ?

Ce n'est **pas un temps en secondes** : c'est un **nombre de coups maximum** (`n_coups_max`, 500 par défaut). Chaque coup joué, bon ou faux, compte pour 1.

- Sans limite, un mauvais agent pourrait retenter 10 000 fois des coups faux (−1, grille inchangée) : la partie ne finirait jamais.
- Au dernier coup autorisé, si la grille n'est pas finie, la partie est coupée et ce coup rapporte **0** au lieu de +1 ou −1.

Exemple : 50 cases vides, limite 70 coups. Un bon joueur remplit en 50 coups : victoire (+11 au dernier coup). Un joueur qui se trompe 30 fois n'a posé que 40 chiffres au 70ᵉ coup : timeout.

## Pourquoi le test devait couvrir le timeout

Le timeout est une règle spéciale : le lot doit la reproduire comme `SudokuEnv`. Première passe avec `n_coups_max = 150` : 256 victoires, **0 timeout**. Le « 0 écart » ne prouvait rien sur cette règle. Avec 70 coups : 10 victoires et 246 timeouts, toujours 0 écart. Les deux cas sont couverts.

Leçon de méthode : un test qui passe n'est utile que s'il exerce le cas qu'on veut vérifier.

## Ce qu'on en déduit

- Un environnement lisible (une grille) et un environnement rapide (un lot) sont deux outils : on garde le premier comme arbitre du second.
- Mesuré : lot sur GPU ≈ 23 fois plus rapide que boucle sur `SudokuEnv`, ≈ 6 fois sur CPU.
- Prochaine leçon : REINFORCE, qui utilise ce lot pour apprendre.
