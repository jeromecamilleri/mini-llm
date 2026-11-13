# 01 — L'environnement et l'oracle

Script : [`../scripts/01_environnement.py`](../scripts/01_environnement.py)
Outils : [`sudoku_env.py`](../tools/sudoku_env.py), [`oracle.py`](../tools/oracle.py), [`sudoku_utils.py`](../tools/sudoku_utils.py)

## L'essentiel en 30 secondes

- En apprentissage par renforcement (RL), le réseau n'apprend pas sur un texte fixe : il **joue**. À chaque tour il voit un **état**, choisit une **action**, et reçoit une **récompense**.
- Ici : l'état est la grille 9 × 9, l'action est « poser le chiffre `d` dans la case `(ligne, colonne)` » (729 actions possibles), la récompense dit si c'était bien.
- L'**oracle** est le seul à connaître la solution. Il répond `OUI` ou `NON`. Le réseau, lui, ne la voit jamais.
- Un **épisode** = une partie, de la grille de départ jusqu'à la victoire ou au timeout.
- Le script joue quelques coups à la main pour voir les récompenses : `+1`, `−1`, `+11` à la victoire.

## Les mots à connaître

| Mot | Sens ici | Équivalent dans le GPT (`04_training`) |
|---|---|---|
| état | la grille courante, 9 × 9 entiers (0 = case vide) | une fenêtre de tokens |
| action | un entier de 0 à 728 : `ligne × 81 + colonne × 9 + (chiffre − 1)` | le token à prédire |
| récompense | un nombre renvoyé après chaque action | la perte (mais on la **maximise**) |
| épisode | une partie complète, de `reset()` à la fin | un passage sur le corpus |
| oracle | juge qui connaît la solution | la cible `y` du corpus |
| case fixe | chiffre donné au départ, jamais modifiable | — |

## Les actions

729 = 81 cases × 9 chiffres. Exemple : poser **4** en ligne 0, colonne 2 :

```
action = 0 × 81 + 2 × 9 + (4 − 1) = 21        decode_action(21) → (0, 2, 4)
```

## L'oracle

Il applique dans l'ordre :

1. case déjà remplie → `NON` ;
2. le chiffre crée un doublon dans la ligne, la colonne ou le bloc 3 × 3 → `NON` ;
3. le chiffre n'est pas celui de la solution → `NON` ;
4. sinon → `OUI`.

> **Écart avec le cahier des charges.** Le cahier prévoyait trois réponses (`OUI`, `NON`, `PEUT_ETRE` pour « légal mais pas la bonne réponse », récompense −0,1). Le code n'en garde que deux : un signal binaire est plus net pour apprendre. Conséquence : un chiffre légal mais faux coûte autant (−1) qu'un doublon. Ce choix pourra être rediscuté en Phase 3 (reward shaping).

## Les récompenses

| Événement | Récompense |
|---|---|
| `OUI` (bon chiffre, case posée) | **+1,0** |
| `NON` (doublon, mauvais chiffre, case déjà remplie) | **−1,0**, la grille ne change pas |
| grille complète et valide | **+10,0** s'ajoute au +1 du dernier coup, soit **+11,0** |
| timeout (`n_coups_max` atteint) | **0,0** |

Le timeout vaut 0 et non une pénalité : le réseau n'est pas puni de « mourir », on arrête proprement. `n_coups_max` vaut 500 par défaut, très au-dessus des 81 coups maximum, pour ne pas décourager l'exploration.

## Exemple suivi (grille « 530070000… », 51 cases vides)

| Coup | Verdict | Récompense | Cases restantes |
|---|---|---|---|
| 7 en (0,1) : case fixe (déjà 3) | NON | −1,0 | 51 |
| 7 en (0,2) : mauvais chiffre | NON | −1,0 | 51 |
| 4 en (0,2) : bon chiffre | OUI | +1,0 | 50 |
| 6 en (0,3) : bon chiffre | OUI | +1,0 | 49 |
| 6 en (0,4) : 6 déjà en ligne 0 | NON | −1,0 | 49 |

Un agent parfait remplit les 51 cases en 51 coups : 50 × (+1) + 11 = **+61,0** (c'est la valeur affichée à l'étape 5).

## L'état encodé pour le réseau (« one-hot »)

Un réseau ne comprend pas « le chiffre 7 » comme un nombre ordinaire : 7 n'est pas « plus » que 6. On donne donc **10 canaux** par case, dont un seul vaut 1 :

| Canal | 0 | 1 | … | 9 |
|---|---|---|---|---|
| signifie | case vide | contient 1 | … | contient 9 |

Forme : `(9, 9, 10)`, `float32`. La somme sur les canaux vaut 1 partout (vérifié à l'étape 6). Ce format sert en Phase 2 à un CNN (ou à un MLP après aplatissement).

## Ce qui est vérifié

| Critère (cahier des charges §10) | Résultat |
|---|---|
| `est_valide` vrai sur une grille correcte, faux avec un doublon | oui |
| oracle : `NON` sur doublon, `OUI` sur le bon chiffre | oui |
| oracle : `PEUT_ETRE` | **supprimé volontairement** |
| `step()` : terminé et +11 sur la dernière case | oui |
| `step()` : terminé au timeout, récompense 0 | oui |
| le script tourne sans GPU ni téléchargement | oui |

## Ce qu'on en déduit

- Un problème de RL tient en trois objets : un **état**, des **actions**, une **récompense**. Tout le reste (réseau, optimiseur) vient après.
- La qualité du signal compte autant que le réseau : si l'oracle est ambigu, le réseau apprend n'importe quoi. D'où le choix binaire.
- Le réseau ne verra que l'état. Les solutions restent dans l'oracle : il doit **découvrir** les règles par essais et erreurs.
- Un agent qui joue au hasard parmi les actions sur cases vides n'a presque jamais raison. Mesuré sur 100 grilles faciles (mini-dataset, `n_coups_max = 200`) : **récompense moyenne −155 par épisode, 0 grille résolue**. C'est la référence à battre en Phase 2.
