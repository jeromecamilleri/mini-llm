# 03 — Mesurer avant d'apprendre

Script : [`../scripts/03_mesures.py`](../scripts/03_mesures.py)
Outil : [`evaluation.py`](../tools/evaluation.py)

## L'essentiel en 30 secondes

- Avant d'entraîner un réseau, on joue des agents **sans apprentissage** pour connaître le point de départ et le plafond.
- Le **masque** liste les actions autorisées (case vide, pas de doublon). Il ne dépend que des règles, pas de la solution : le réseau pourra l'utiliser. Il réduit 729 actions à environ 160 au départ, soit environ 3,3 chiffres possibles par case vide.
- Découverte : un coup faux ne modifie pas la grille. Avec le masque, **le hasard résout donc toutes les grilles**. Le taux de résolution ne sert plus de mesure ; on mesure le **nombre de coups** et le **retour**.
- Jalons : hasard + masque (départ, environ 90 coups) → case contrainte (méthode humaine, environ 55) → parfait (le nombre de cases vides, environ 52).

## Définitions

| Mot | Sens |
|---|---|
| politique | fonction qui choisit l'action à partir de l'état (plus tard : le réseau) |
| masque | tableau de 729 booléens : l'action est-elle autorisée ? |
| retour | somme des récompenses d'un épisode |
| parfait | un coup juste par case vide : retour = cases vides + 10 (ex. 51 cases → +61) |

## Les quatre agents

1. **hasard** : une action parmi les 729.
2. **hasard + masque** : une action parmi les autorisées.
3. **case contrainte** : la case vide qui a le moins de chiffres autorisés, puis un chiffre autorisé au hasard. Exemple : si une case n'accepte que le 4, le coup est sûr. Il n'utilise que les règles.
4. **parfait** : calculé, pas joué.

## Les résultats (200 grilles par niveau, `n_coups_max` = 500, mini-dataset)

| Niveau | Agent | Résolues | Retour | Coups justes | Coups pour finir |
|---|---|---|---|---|---|
| facile | hasard | 0 % | −447,6 | 5 % | — |
| facile | hasard + masque | 100 % | +25,0 | 59 % | 87 |
| facile | case contrainte | 100 % | +59,6 | 97 % | 52 |
| facile | parfait | 100 % | +61,0 | 100 % | 51 |
| diabolique | hasard | 0 % | −446,3 | 5 % | — |
| diabolique | hasard + masque | 100 % | +22,5 | 57 % | 94 |
| diabolique | case contrainte | 100 % | +58,1 | 91 % | 59 |
| diabolique | parfait | 100 % | +63,4 | 100 % | 53 |

(moyen et difficile sont dans la sortie du script : valeurs intermédiaires, hasard + masque 93 à 94 coups, case contrainte 57 à 58.)

## Ce qu'on voit

- **Sans masque, le hasard ne gagne jamais** : 5 % de coups justes, retour d'environ −447 sur 500 coups.
- **Avec masque, le hasard gagne toujours**, mais lentement : environ 1,7 fois le minimum de coups.
- **La méthode humaine de base** (case la plus contrainte) approche le minimum, même sur « diabolique » : 91 % de coups justes. Aucun niveau ne l'arrête.
- **La difficulté de la source ne change presque rien** pour ces agents : les notes (`rating`) mesurent la difficulté pour un humain qui applique des techniques, pas pour un agent qui essaie des chiffres.

## Pourquoi le taux de résolution ne suffit plus

L'environnement est **indulgent** : un coup faux est puni (−1) mais la grille reste correcte, il n'y a pas de coup irréversible. Un agent qui insiste finit toujours par trouver. Ce que le réseau peut gagner n'est donc pas « résoudre » mais **résoudre en moins de coups**, de 90 vers 52. La mesure est : retour moyen et nombre de coups à `n_coups_max` fixé.

Conséquence pour les décisions à venir (à trancher à la leçon 06) :
- **Masque** : il fait presque tout le travail. Sans lui, on ne peut pas entraîner en 30 à 60 min.
- **Réseau** : il n'a rien d'autre à apprendre que « quel chiffre dans quelle case parmi les autorisées ». L'agent « case contrainte » (91-97 %) montre le plafond d'une méthode simple.
- **`PEUT_ETRE`** : reste à tester, sans raison de le préférer pour l'instant (voir `docs/suivi_sudoku_rl.md`).

## Ce qu'on en déduit

- Une mesure se choisit **après** avoir regardé ce que les agents simples font, pas avant : « taux de grilles résolues » aurait été trompeur dès qu'on masque.
- Le départ à battre : environ 90 coups, retour +23 à +25. Le plafond : environ 52 coups, retour +61.
- Le masque est un choix de conception, pas de l'apprentissage : il faudra le dire quand on présentera un taux de réussite du réseau.
