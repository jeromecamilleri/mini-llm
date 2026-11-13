# Spécification — Environnement B (sans oracle)

> **Auteur : Claude Sonnet 5.5 (identifiant de modèle : `claude-sonnet-5.5`)**, via GitHub Copilot CLI, le 13 novembre 2026.
> **Statut : spécification. Aucun code n'est écrit, aucun fichier d'implémentation n'est créé.**
> Origine : guide de décision `docs/audits/guide_decisions_experiences_agent_sudoku_rl_2026-11-13.md` (§3.2, §3.3, §8 Phase 1), et sept consignes du porteur du projet, reprises ci-dessous (R1 à R6 et §7).

L'environnement **A** (avec oracle) reste l'environnement d'entraînement actuel et n'est pas modifié. L'environnement **B** décrit ici sert à **mesurer l'autonomie** : l'agent joue sans que personne ne lui dise si un chiffre légal est le bon. Les résultats de A et de B ne sont **jamais agrégés** dans un score unique.

## 1. Ce que B est, et ce qu'il n'est pas

| B est | B n'est pas |
|---|---|
| Un environnement d'évaluation de la résolution autonome | Un environnement d'entraînement (décision différée, §7.3) |
| Une copie des règles de A : mêmes actions, même masque, même observation | Une modification de A : `SudokuEnvLots` et `SudokuEnv` restent intacts |
| Un jeu où une erreur a des conséquences réelles | Un jeu où l'on peut réessayer sans coût |

## 2. Règles de l'environnement (R1 à R6)

### R1. Une action localement légale est toujours inscrite

L'agent choisit une action `(case, chiffre)`, codée comme dans A : `ligne × 81 + colonne × 9 + (chiffre − 1)`, soit 729 actions.

Une action est **localement légale** si la case est vide et si le chiffre est absent de sa ligne, de sa colonne et de son bloc 3 × 3. C'est exactement le masque de A (`reseau.py:masques_actions`, `env_lots.py:masques`).

Toute action légale est **inscrite dans la grille**, qu'elle soit correcte ou non par rapport à la solution. Personne ne vérifie l'action avec une solution pendant la partie.

Une action illégale **n'est pas un coup** : elle déclenche une erreur d'interface (exception). Les politiques de B sont toujours masquées ; une action illégale signale un défaut d'implémentation, pas une erreur de jeu.

### R2. Une contradiction termine immédiatement la partie en échec

Après chaque inscription, l'environnement teste le critère local suivant : **existe-t-il une case vide qui n'a aucun chiffre légal ?** Si oui, la partie se termine **immédiatement en échec** (raison : `contradiction`).

Ce critère ne dépend que de la grille et des règles. Il ne consulte pas la solution.

Pourquoi terminer tout de suite : les chiffres ne sont jamais retirés (R4), donc l'ensemble des chiffres légaux d'une case ne peut que **diminuer**. Une case vide sans chiffre légal le restera : la contradiction est **irrécupérable**. La continuer ne peut pas transformer l'échec en succès.

Cas particulier : s'il reste des cases vides mais qu'aucune action légale n'existe, alors chaque case vide est sans chiffre légal : c'est une contradiction. Le cas « plus aucune action possible » est donc inclus dans R2 et n'est pas une condition terminale séparée.

**Limite volontaire** : l'environnement ne détecte pas les impasses plus subtiles, par exemple un chiffre qui n'a plus aucune case possible dans une ligne alors que chaque case vide a encore un candidat. Détecter de telles situations serait une aide experte (propagation de contraintes), exclue par le guide. Ces impasses se terminent plus tard par une contradiction de type R2.

### R3. Une grille complète est évaluée : succès uniquement si c'est une solution valide du Sudoku initial

Quand plus aucune case n'est vide, l'environnement évalue la grille. Il y a **succès si et seulement si** :
1. aucune case n'est vide ;
2. chaque ligne, colonne et bloc 3 × 3 contient les chiffres 1 à 9 exactement une fois ;
3. les chiffres donnés au départ sont inchangés (compatibilité avec le Sudoku initial).

Ce test **n'utilise pas la solution stockée** : il ne dépend que de la grille initiale et des règles. Il reste correct même si une grille de départ avait plusieurs solutions. Les 512 grilles de test et les 1 000 du mini-pool ont chacune exactement une solution (vérifié avec un compteur de solutions ; il en trouve ≥ 2 sur 20 grilles appauvries de contrôle). Pour ces grilles, une solution valide du Sudoku initial **est** la solution stockée.

Conséquence à vérifier par test (§9) : comme toute inscription est légale (R1) et que les chiffres donnés ne sont jamais touchés, **une grille complète construite par B est toujours valide**. L'agent ne peut donc pas terminer sur une grille complète fausse : une erreur se paie par une contradiction (R2) ou, au pire, par la limite de coups (R5).

### R4. Aucun retour arrière, aucune correction automatique

- Pas d'action « effacer » ni « annuler ».
- Pas de modification d'une case remplie, y compris les cases données au départ.
- Pas de réparation automatique, pas de propagation de contraintes, pas de remplissage des cases forcées (chiffres uniques) par l'environnement.
- Pas d'indice ni de suggestion.

### R5. Une limite de coups évite les épisodes sans fin

- **Limite : 81 actions** (nombre de cases de la grille), paramètre explicite `n_coups_max`.
- Au-delà, la partie se termine en échec (raison : `limite`).

Chaque action inscrit un chiffre dans une case vide (R1), donc une partie comporte **au plus autant d'actions que de cases vides au départ** (≈ 50 à 54 sur nos grilles). La limite est **un filet de sécurité qui ne devrait jamais être atteint**. Tout épisode terminé par `limite` doit être compté séparément et traité comme un **signal d'alerte** (défaut d'implémentation ou de la spécification).

### R6. L'oracle ne sert qu'à constituer les données et à évaluer le résultat final

L'oracle (la solution connue) n'intervient qu'à deux endroits :
1. **Constitution des données** : le solveur local calcule la solution des grilles du dataset (`dataset.py:construire_dataset`, `sudoku_utils.py:resoudre`).
2. **Évaluation a posteriori** : après la fin de la partie, l'évaluateur compare la grille finale à la solution pour produire des diagnostics (§6).

Il **ne guide jamais l'agent pendant la partie**. La règle est imposée par la structure : l'environnement B est construit **avec le seul puzzle**, sans la solution. Il ne peut donc pas en fuir.

| Interdit pendant la partie | Autorisé |
|---|---|
| Verdict OUI/NON ou toute récompense calculée depuis la solution | Observation : grille courante, 10 canaux comme dans A |
| Information sur le chiffre correct d'une case | Masque légal calculé depuis la grille et les règles |
| Mémoire des actions refusées (il n'y en a pas) | Indication de fin de partie |
| Annonce d'une contradiction avant la fin de partie (la contradiction **est** la fin) | Le nombre d'actions écoulées n'est pas fourni à l'agent |

## 3. Interface (description, sans code)

| Élément | Contenu |
|---|---|
| Construction | le **puzzle** (9 × 9, 0 = vide) ; `n_coups_max` (81 par défaut) ; **jamais la solution** |
| Observation | la grille courante, encodée comme dans A (10 canaux : vide, 1 à 9), rien d'autre |
| Masque | actions localement légales, calculé depuis la grille (même fonction que A) |
| Étape | une action légale → nouvelle observation + indicateur « terminé » ; aucune récompense dérivée de la solution |
| Fin de partie | une raison exposée **après** la fin : `succes`, `contradiction` ou `limite` |
| Bilan évaluateur | fonction séparée, reçoit la grille finale et la solution, **hors du chemin de décision de l'agent** |
| Lots | comme `SudokuEnvLots`, plusieurs grilles en parallèle ; une partie terminée est gelée |

Aucune fonction de l'environnement B ne renvoie à l'agent un champ dérivé de la solution.

## 4. Conditions terminales

| Raison | Condition | Succès ? |
|---|---|---|
| `succes` | grille complète, valide, compatible avec le puzzle initial (R3) | **oui** |
| `contradiction` | après une inscription, une case vide n'a aucun chiffre légal (R2) | non |
| `limite` | 81 actions atteintes sans autre fin (R5) | non, et signal d'alerte |

Chaque partie se termine par **exactement une** raison. La vérification de R2 précède celle de R3 : une grille complète n'a aucune case vide, donc ne peut pas contenir de contradiction.

## 5. Propriétés déduites (à vérifier par tests avant tout usage)

1. **Monotonie** : les chiffres légaux d'une case vide ne peuvent que diminuer au cours de la partie.
2. **Contradiction irrécupérable** : une fois R2 vérifiée, aucune suite d'actions ne peut mener au succès.
3. **Complet ⇒ valide** : toute grille complète produite par B est valide et compatible avec le puzzle.
4. **Fin garantie** : une partie dure au plus `nombre de cases vides` actions ; `limite` n'est jamais atteinte avec le masque actuel.
5. **L'argmax est bien défini** : chaque action modifie la grille, donc il n'y a pas de répétition d'un même état. Dans A, jouer l'argmax échoue (0 % résolues) parce que le même coup faux se répète ; ce problème disparaît dans B.
6. **Identité des règles avec A** : à grille égale, les masques de A et de B sont identiques.

*Indice, non versionné : lors d'une mesure exploratoire (§8), la part de grilles « complètes » et celle de grilles « exactes » étaient identiques pour tous les agents (29,0 % et 29,0 % pour MRV, 0 % et 0 % pour les autres), ce qui est cohérent avec la propriété 3.*

## 6. Évaluation et métriques

### 6.1 Métriques principales (guide §3.2)

| Métrique | Définition |
|---|---|
| Succès | part des grilles terminées en `succes` |
| Contradiction | part des grilles terminées en `contradiction` |
| Limite | part des grilles terminées en `limite` (doit être 0) |
| Actions inscrites | nombre d'actions jouées jusqu'à la fin (= cases remplies par l'agent) |
| Cases correctes | parmi elles, celles dont le chiffre égale la solution (évaluateur, a posteriori) |
| Cases fausses | actions inscrites − cases correctes |
| Part du travail faite | cases correctes / cases vides au départ |

### 6.2 Diagnostics (évaluateur seul, sans effet sur l'agent)

- **Première erreur** : rang (à partir de 1) de la première action dont le chiffre n'est pas celui de la solution ; vide si aucune.
- **Latence de contradiction** : nombre d'actions entre la première erreur et la fin, pour les échecs. Elle mesure à quel point une erreur reste « cachée » avant de produire une contradiction.
- **Résultats par difficulté** (facile, moyen, difficile, diabolique) et par graine d'entraînement.

### 6.3 Protocole

- **Mêmes 512 grilles de test** que pour A (`logs/sudoku_test_ids.csv`), **mêmes checkpoints** (10 canaux ; un checkpoint à 19 canaux est refusé), même noyau de graines.
- **Deux modes de jeu**, rapportés séparément : tirage selon les probabilités, et argmax (propriété 5).
- Résultats **grille par grille et par graine**, comme pour A. Plusieurs tirages sur une même grille ne sont pas des grilles indépendantes ; l'incertitude entre entraînements (4 graines au minimum) est rapportée séparément de l'incertitude de tirage.
- **Fichiers de sortie séparés** de ceux de A (par exemple `logs/sudoku_reference_B_*`), jamais fusionnés.
- Pour un succès, une incertitude binomiale sur 512 grilles s'ajoute à l'écart entre entraînements ; ne pas en tirer d'effet sans les deux.
- Les nombres de A (coups faux, répétitions) n'ont **pas d'équivalent** dans B et ne doivent pas être comparés à ceux de B.

### 6.4 Agents de référence (sans apprentissage)

| Agent | Description |
|---|---|
| Hasard plat | action uniforme parmi les actions légales |
| Hasard factorisé | case uniforme parmi celles qui ont un chiffre légal, puis chiffre légal uniforme |
| MRV qui devine | case au plus petit nombre de chiffres légaux, puis chiffre légal au hasard ; ne devine que si aucun chiffre unique n'existe |

MRV est une **référence externe**, jamais une cible d'apprentissage. Un agent « chiffres uniques seulement » ne peut pas exister dans B (il faut toujours jouer une action tant que la partie continue) ; sa mesure est réservée à une analyse a posteriori.

## 7. Récompenses et conditions terminales (documentées à part, non implémentées)

### 7.1 Évaluation de B

**Aucune récompense n'est utilisée pour évaluer.** B produit des métriques (§6), pas un signal d'apprentissage. Les récompenses de A (+1, −1, +10, 0) **ne s'appliquent pas** à B.

### 7.2 Si un entraînement dans B était décidé un jour

C'est une décision différée (guide §10, question ouverte). Si elle était prise, les contraintes seraient :

| Contrainte | Raison |
|---|---|
| Récompense définie **uniquement** à partir des conditions terminales de §4 | pas de récompense issue de la solution pendant la partie |
| Pas de bonus heuristique (cases à chiffre unique, progression, MRV) | guide §2.1 : aucune connaissance experte |
| Récompense de B jamais additionnée à celle de A | deux évaluations distinctes |

Deux formulations candidates, **à ne pas retenir aujourd'hui** :

| Candidate | Récompense en cours de partie | Récompense finale |
|---|---|---|
| Binaire | 0 | +1 succès, 0 sinon |
| Binaire avec pénalité | 0 | +1 succès, −1 contradiction ou limite |

Risque documenté : avec des politiques actuellement à ≈ 0 % de succès, un signal binaire est presque toujours nul (les mesures préliminaires donnent 0 % pour les réseaux). L'article AdaC-GRPO (arXiv:2610.13184), sur un autre type de modèle, décrit le même phénomène de récompense rare à succès quasi nul et y répond par un curriculum ; ce résultat n'est pas transposable tel quel et n'est cité que comme piste à étudier.

### 7.3 Statut

Aucune récompense de B n'est décidée ni implémentée. Toute décision à ce sujet exige une expérience isolée, une hypothèse explicite et l'accord du porteur du projet.

## 8. Mesures préliminaires (exploratoires, non versionnées)

Mesurées dans une session précédente par des commandes ponctuelles, **sans script versionné** : 512 grilles de test, les coups légaux inscrits tels quels. La règle de fin était « s'arrêter quand plus aucun coup légal », différente de R2 ; le taux de succès n'en est pas affecté (la contradiction est irrécupérable, §2 R2), mais les chiffres de cases fausses ne le sont pas non plus de façon comparable.

| Agent | Grilles exactes | Remarque |
|---|---:|---|
| Hasard plat | 0 % | 4 tirages |
| Hasard factorisé | 0 % | 4 tirages |
| MRV qui devine | 29,0 % | 4 tirages |
| Chiffres uniques seuls, puis arrêt | 18,4 % | borne de ce que permettent les seules cases forcées |
| Réseaux de référence g0 à g3 | 0 % | argmax et tirage |

À reproduire par un script versionné avant toute utilisation comme référence.

## 9. Tests à écrire avant d'utiliser B

1. Une action légale mais fausse est inscrite ; la grille change.
2. Une action légale et correcte est inscrite, de la même façon.
3. Une action illégale lève une erreur et ne modifie rien.
4. La contradiction est détectée dès qu'une case vide n'a plus de chiffre légal ; la partie se termine en `contradiction`.
5. L'absence totale d'action légale avec des cases vides est classée `contradiction`.
6. Une grille complète valide est classée `succes` ; le test n'utilise pas la solution.
7. Une grille complète invalide, construite à la main hors de l'interface normale, est classée échec (défense en profondeur de R3).
8. `limite` est atteinte avec un masque artificiellement forcé, et jamais avec le masque normal.
9. Une partie terminée est gelée et ignore les actions suivantes.
10. À grille égale, masques de A et de B identiques (comparaison sur des états intermédiaires, pas seulement initiaux).
11. L'environnement se construit sans solution et aucune de ses sorties ne contient de champ dérivé de la solution.
12. Sur les 512 grilles de test et le mini-pool, une grille complète produite par B égale la solution stockée (corollaire de l'unicité).
13. Comparaison avec A : sur des suites d'actions **toutes correctes**, B et A donnent les mêmes grilles ; toute divergence ne vient que des erreurs.

## 10. Questions ouvertes

| Question | Statut |
|---|---|
| Valeur exacte de `n_coups_max` | proposée à 81, à confirmer (jamais atteinte en pratique) |
| Faut-il fournir à l'agent le nombre d'actions restantes ? | non, pour garder la même observation que A ; à rediscuter seulement par expérience isolée |
| Entraîner dans B ? | différé ; voir §7 |
| Évaluer aussi les checkpoints à l'argmax dans A ? | hors du périmètre de B |
| Nom et emplacement des fichiers d'implémentation | proposition : `tools/env_sans_oracle.py` (classe séparée de `SudokuEnvLots`) et `scripts/07_evaluation_B.py`, **non créés** |

## 11. Contraintes respectées

Aucune supervision experte, aucun label de solution ni de candidats, aucune imitation d'un solveur, aucune mémoire des refus, aucune tête auxiliaire. Le masque est celui des règles locales, sans solution. L'objectif générique du projet est inchangé.
