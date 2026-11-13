# 02 — Le dataset de grilles

Script : [`../scripts/02_dataset.py`](../scripts/02_dataset.py)
Outils : [`dataset.py`](../tools/dataset.py), [`sudoku_utils.py`](../tools/sudoku_utils.py) (solveur, validation)

## L'essentiel en 30 secondes

- Le dataset fournit des **problèmes** à l'environnement : chaque ligne = une grille de départ (`puzzle`), sa solution, un niveau et une note.
- Source : [sudoku-exchange-puzzle-bank](https://github.com/grantm/sudoku-exchange-puzzle-bank) (4 fichiers bruts : `easy`, `medium`, `hard`, `diabolical`), qui donnent le puzzle et sa note, **mais pas la solution**.
- Notre **solveur** (backtracking) calcule les solutions, puis chaque paire est validée.
- Résultat : `sudoku_dataset.csv` (**200 000 grilles**, 50 000 par niveau, 35 Mo) et `sudoku_dataset_mini.csv` (**1 000 grilles**, 250 par niveau) pour développer vite.
- Ces fichiers sont dans `.gitignore` (`datasets/sudoku/*`) : on les reconstruit avec le script.

## Le format

```
puzzle,solution,difficulte,rating
0507030600078000008168…,158723469367954821294816…,facile,1.2
```

| Colonne | Contenu |
|---|---|
| `puzzle` | 81 caractères, ligne par ligne, `0` = case vide |
| `solution` | 81 chiffres, sans zéro |
| `difficulte` | `facile`, `moyen`, `difficile` ou `diabolique` |
| `rating` | note de difficulté d'origine (échelle de « Sudoku Explainer ») |

`SudokuDataset` renvoie pour `ds[i]` un dictionnaire : `puzzle` et `solution` en `(9, 9)` `int8`, `difficulte`, `rating`, `id`.

## Le solveur

Il sert uniquement à fabriquer les solutions (jamais à jouer à la place du réseau). Il combine :

- **MRV** : on traite toujours la case qui a le moins de candidats ;
- **propagation** : poser un chiffre retire ce candidat des cases de la même ligne, colonne et bloc ; une case à un seul candidat est posée en cascade ;
- **retour arrière** : en cas d'impasse, on restaure l'état et on essaie un autre candidat.

## La validation de chaque paire

Une grille est rejetée (avec un avertissement) si : longueur ≠ 81, zéro dans la solution, case donnée différente de la solution, ou doublon dans la solution. Contrôle fait pour cette fiche : 5 000 grilles du CSV complet re-validées, **0 erreur**.

## La difficulté : un écart avec le cahier des charges

Le cahier proposait 3 niveaux définis par le **nombre de cases vides** (25-35, 36-46, 47-58). Le code utilise à la place les **4 fichiers sources** et leur note. Mesuré sur les 200 000 grilles :

| Niveau | Note (rating) | Cases vides : min | max | moyenne |
|---|---|---|---|---|
| facile | 1,2 (toutes) | 37 | 60 | 50,7 |
| moyen | 1,5 à 2,3 | 44 | 59 | 53,2 |
| difficile | 2,5 à 4,8 | 44 | 59 | 53,2 |
| diabolique | 5,0 à 9,2 | 44 | 59 | 53,4 |

Ce qu'on voit :

- **Toutes les grilles ont environ 50 à 54 cases vides**, quel que soit le niveau : le nombre de cases vides ne mesure **pas** la difficulté. Une grille à 55 cases vides peut être facile (chaque case se déduit d'un coup) ou diabolique (il faut une technique avancée).
- Le niveau `facile` ne contient que la note 1,2 (grilles qui se résolvent uniquement avec des « singletons » : une case n'accepte plus qu'un chiffre, ou un chiffre n'a plus qu'une place dans une ligne, colonne ou bloc).
- Les écarts de cases vides entre niveaux sont donc minimes : pour un curriculum (Phase 3), on triera plutôt sur le `rating`.

Le cahier des charges n'est pas à jour sur ce point ; le code, lui, est cohérent avec les données.

## Les opérations disponibles

| Appel | Effet |
|---|---|
| `par_difficulte("facile")` | vue filtrée |
| `melanger(graine=0)` | réordonne, de façon reproductible (même graine = même ordre) |
| `sous_echantillon(n, graine)` | n grilles au hasard |
| `distribution()` / `stats_cases_vides()` | comptes et statistiques par niveau |

## Durées

| Étape | Durée |
|---|---|
| lecture et validation du mini (1 000 grilles) | quelques secondes |
| construction complète (200 000 grilles, solveur) | selon le CPU, de l'ordre de dizaines de minutes |

Le script ne reconstruit pas un CSV déjà présent (supprimer le fichier pour forcer).

## Ce qu'on en déduit

- Pour un RL, « le dataset » ne sert pas à apprendre directement : il fournit des **situations de départ**. L'apprentissage vient des récompenses.
- Une solution unique par grille est indispensable : l'oracle dit `NON` à tout autre chiffre, même s'il respecte les règles. Les grilles du dépôt source garantissent cette unicité par construction.
- Vérifier les données avant d'apprendre (écart de niveaux, validité) évite de chercher une erreur dans le réseau qui est en fait dans les données.
- Pour le curriculum, la bonne mesure de difficulté est la note, pas le nombre de cases vides.
