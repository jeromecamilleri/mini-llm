# Plan de reprise — entraîner un agent Sudoku générique, sans labels ajoutés

> Spécification destinée à une IA qui reprend le développement du projet.  
> Dépôt : `jeromecamilleri/mini-llm`  
> Périmètre : `src/07_sudoku_rl/`  
> Décision de conception : **ne pas mettre en œuvre l’expérience 16-A** (mémoire des refus / canaux supplémentaires).

---

## 1. Intention et objectif à respecter

L'objectif initial est de faire apprendre à un modèle à jouer au Sudoku de la manière la plus générique possible.

Le modèle doit apprendre à partir :
- de l'état courant de la grille ;
- des règles intrinsèques du Sudoku ;
- des conséquences de ses actions dans l'environnement ;
- du signal de récompense déjà prévu par le projet.

**Ne pas ajouter de labels artificiels ou de connaissances expertes dans les données d'entraînement.** En particulier :
- ne pas fournir au modèle le chiffre solution d'une case comme cible supervisée ;
- ne pas lui fournir une étiquette « bonne case à choisir » ;
- ne pas lui fournir une liste de candidats annotée par un solveur externe comme vérité terrain d'apprentissage ;
- ne pas lui apprendre explicitement la stratégie MRV ;
- ne pas ajouter de canaux de mémoire des refus à l'entrée du réseau ;
- ne pas entraîner une tête auxiliaire à prédire les solutions ou les candidats.

Le modèle doit découvrir une stratégie à partir des règles et de l'expérience, pas imiter un solveur qui lui fournirait les réponses.

### Distinction essentielle : règles, masque et labels

Utiliser les règles du Sudoku pour déterminer si une action est légalement possible n'est pas ajouter un label d'apprentissage. C'est une contrainte de l'environnement, comme les règles d'un jeu.

En revanche, fournir au réseau une cible calculée à partir de la solution — par exemple « pour cette case, le bon chiffre est 7 » — serait une supervision supplémentaire. Cette supervision est volontairement exclue de ce plan.

Le masque d'actions légales peut donc être conservé et amélioré. Il ne doit utiliser que l'état visible de la grille et les règles standard : unicité des chiffres dans chaque ligne, colonne et bloc 3×3.

L'oracle qui connaît la solution peut rester dans l'environnement pour attribuer la récompense après une action, conformément au fonctionnement actuel. Il ne doit pas exposer la solution au réseau ni générer des labels supervisés.

---

## 2. Décision : abandonner 16-A

### 16-A — mémoire des actions refusées : NON

Ne pas développer cette variante dans cette phase.

Cela signifie :
- ne pas ajouter de mémoire des actions refusées dans l'observation ;
- ne pas ajouter neuf canaux au tenseur d'entrée ;
- ne pas modifier le réseau pour lui faire apprendre l'historique des refus ;
- ne pas consacrer une expérience spécifique à cette fonctionnalité.

Pourquoi ?

1. Cela ajoute une information spécifique à l'implémentation et éloigne le modèle d'une représentation générique de l'état Sudoku.
2. Cela augmente la complexité de l'environnement et de l'entrée du réseau.
3. Cela risque d'améliorer les résultats en évitant des répétitions sans améliorer réellement la compréhension des contraintes.
4. Ce n'est pas nécessaire pour tester l'hypothèse principale : une représentation d'action mieux structurée peut-elle faciliter l'apprentissage ?

Une action refusée peut être réessayée par la politique si le modèle n'a pas appris à l'éviter. C'est un coût de l'apprentissage par expérience. On mesurera ce phénomène, mais on ne le corrigera pas par une mémoire dédiée dans cette phase.

---

## 3. Point de départ : conserver une référence fiable

Avant toute modification :

1. Conserver le code et les poids de l'agent actuel comme référence.
2. Identifier précisément le script d'entraînement, le réseau, l'environnement, l'oracle et les fonctions de masque.
3. Relever la configuration complète d'un entraînement reproductible.
4. Fixer un jeu de grilles de test indépendant de l'entraînement.
5. Exécuter plusieurs fois l'agent actuel avec les mêmes grilles et plusieurs graines.
6. Enregistrer les métriques et les journaux.

Ne pas modifier simultanément l'environnement, la représentation, l'algorithme et les récompenses. Il faut pouvoir attribuer les différences observées à une modification identifiée.

### État de référence à documenter

- nombre de paramètres du réseau ;
- forme exacte du tenseur d'entrée ;
- nombre d'actions ;
- définition du masque ;
- fonction de récompense ;
- algorithme d'entraînement et hyperparamètres ;
- taille des lots parallèles ;
- horizon maximal ;
- répartition des difficultés ;
- métriques par difficulté ;
- graines aléatoires et identifiants des grilles de test.

Ne pas reprendre des moyennes issues de logs différents sans vérifier que les protocoles sont identiques.

---

## 4. Étape principale — 16-B : factoriser l'action sans ajouter de labels

### 4.1 Problème actuel

La politique actuelle choisit une action parmi 729 couples :

\[
81\ \text{cases}\times 9\ \text{chiffres}=729
\]

Elle doit donc apprendre dans une seule sortie :
- quelle case sélectionner ;
- quel chiffre inscrire dans cette case.

Cette représentation ne rend pas explicite la dépendance entre le choix de la case et le choix du chiffre.

### 4.2 Nouvelle représentation proposée

Séparer la politique en deux distributions :

\[
P(c,d\mid s)=P(c\mid s)\,P(d\mid c,s)
\]

avec :
- \(s\) : état courant de la grille ;
- \(c\) : case sélectionnée parmi les 81 cases ;
- \(d\) : chiffre sélectionné parmi les 9 chiffres.

Le réseau comporte donc deux têtes de politique :

1. **Tête case** : 81 logits, un par case.
2. **Tête chiffre** : 9 logits, calculés conditionnellement à l'état et à la case sélectionnée.

La deuxième tête doit réellement être conditionnée par la case choisie. Une possibilité simple est de produire des représentations spatiales par case, puis de récupérer la représentation de la case sélectionnée pour calculer les 9 logits. Éviter de construire une deuxième tête qui ignorerait le choix de case.

### 4.3 Utilisation du masque légal

Le masque reste calculé par l'environnement à partir des règles du Sudoku.

Pour la tête case :
- masquer les cases déjà remplies ;
- masquer les cases pour lesquelles aucun chiffre légal n'existe ;
- ne pas imposer de préférence MRV ou une autre heuristique.

Pour la tête chiffre :
- une fois la case choisie, masquer les chiffres qui créeraient un doublon dans sa ligne, sa colonne ou son bloc ;
- échantillonner parmi les chiffres légalement possibles pour cette case.

Le réseau ne reçoit pas les candidats comme des labels. Le masque sert uniquement à empêcher l'environnement de proposer une action qui viole immédiatement les règles.

Important : ne pas confondre « légal selon les contraintes locales » et « correct selon la solution ». Un chiffre peut être localement légal et néanmoins ne pas être le chiffre de la solution. C'est précisément à l'agent d'apprendre à distinguer ces possibilités à partir de son expérience et du retour de l'environnement.

### 4.4 Politique et calcul de la loss

Pour une action composée de la case \(c_t\) et du chiffre \(d_t\), la log-probabilité de l'action complète est :

\[
\log \pi(a_t\mid s_t)
=
\log \pi_{\text{case}}(c_t\mid s_t)
+
\log \pi_{\text{chiffre}}(d_t\mid c_t,s_t)
\]

La loss de politique doit utiliser cette log-probabilité totale, et non seulement celle de la tête chiffre ou de la tête case.

Conserver dans un premier temps l'algorithme REINFORCE existant et le même signal de récompense. L'objectif de cette expérience est d'isoler l'effet de la factorisation, pas de changer en même temps d'algorithme RL.

### 4.5 Ne pas introduire MRV comme cible

Ne pas entraîner la tête case à reproduire la stratégie MRV et ne pas lui fournir une étiquette calculée par un solveur.

MRV peut être exécuté comme **baseline externe de comparaison**, mais il ne doit pas devenir une cible supervisée du réseau.

Le modèle doit apprendre lui-même quelles cases choisir, en fonction de l'état et des retours de l'environnement.

---

## 5. Étape suivante — entraînement RL avec la nouvelle politique

Une fois la politique factorisée implémentée et testée :

1. Utiliser les mêmes grilles d'entraînement que la référence, ou un jeu contrôlé équivalent.
2. Conserver le même oracle et les mêmes règles de récompense.
3. Conserver les mêmes limites d'horizon.
4. Garder le même algorithme REINFORCE pour cette première comparaison.
5. Entraîner plusieurs graines.
6. Évaluer sur le jeu de test fixe, sans adaptation ni exploration si le protocole de test l'exige.
7. Comparer avec la politique initiale à 729 sorties.

### Récompenses

Ne pas ajouter de nouvelles récompenses heuristiques à ce stade.

Conserver le signal déjà implémenté dans le dépôt, après l'avoir vérifié dans le code :
- récompense d'un placement correct ;
- pénalité d'un placement incorrect ;
- bonus de résolution complète ;
- comportement défini en cas d'expiration de l'horizon.

Ne pas ajouter de bonus pour MRV, pour le nombre de candidats, pour la progression vers une solution ou pour l'imitation d'un solveur. Cela introduirait des connaissances supplémentaires et rendrait l'expérience difficile à interpréter.

---

## 6. Étape complémentaire — curriculum de difficulté, sans labels

Les travaux sur le curriculum, notamment l'approche AdaC-GRPO examinée, suggèrent qu'un agent peut bénéficier d'une progression graduelle de la difficulté.

On peut reprendre cette idée sans ajouter de labels ni de règles expertes dans le réseau.

### Principe

- commencer avec une distribution de grilles relativement faciles ;
- mesurer les performances sur un jeu de validation fixe ;
- augmenter progressivement la proportion de grilles plus difficiles lorsque l'agent atteint un niveau de réussite défini ;
- conserver toujours un échantillon de grilles des niveaux précédents pour éviter l'oubli ;
- ne jamais utiliser les solutions pour guider directement les actions du réseau.

Le curriculum ne modifie ni l'observation ni les sorties du modèle. Il modifie seulement la distribution des épisodes d'entraînement.

### Précautions

- définir les seuils de progression à l'avance ;
- ne pas ajuster le curriculum sur le jeu de test ;
- enregistrer la distribution des difficultés à chaque étape ;
- comparer à un entraînement sans curriculum ;
- éviter de changer simultanément le curriculum et plusieurs hyperparamètres.

Cette étape vient après la comparaison initiale entre politique plate et politique factorisée. Elle doit être évaluée comme une expérience indépendante.

---

## 7. Algorithme : ne pas changer tout de suite REINFORCE

Le réseau actuel possède une tête de valeur, mais le code examiné semble ne pas l'utiliser dans la loss REINFORCE.

Pour la première expérience factorisée :
- conserver REINFORCE ;
- vérifier précisément le calcul des retours et des avantages ;
- vérifier si la tête de valeur est effectivement entraînée ;
- si elle ne sert pas, ne pas prétendre faire de l'actor-critic ;
- éviter de changer simultanément vers PPO ou GRPO.

Après avoir obtenu une comparaison propre, une expérience ultérieure pourra tester une baseline apprise ou un actor-critic. Ce changement devra être isolé et mesuré.

L'objectif est d'abord de savoir si la représentation factorisée améliore l'apprentissage avec le même algorithme.

---

## 8. Baselines et protocole d'évaluation

Comparer au minimum :

1. **Hasard avec masque légal** : référence basse, sans stratégie apprise.
2. **Heuristique MRV** : référence algorithmique externe, non utilisée comme label.
3. **Politique actuelle à 729 actions + REINFORCE** : référence du projet.
4. **Politique factorisée 81 + 9 + REINFORCE** : expérience principale.
5. **Politique factorisée avec curriculum** : expérience complémentaire, seulement après validation de la précédente.

### Mesures à enregistrer

| Mesure | Intérêt |
|---|---|
| Taux de résolution sous horizon fixe | Mesure principale de réussite |
| Nombre de coups jusqu'à résolution | Efficacité |
| Nombre de coups incorrects | Qualité des décisions |
| Nombre de répétitions | Quantifie le coût des essais infructueux |
| Récompense cumulée | Suit l'objectif d'entraînement |
| Résultats par difficulté | Mesure la généralisation |
| Moyenne et dispersion entre graines | Évite de conclure sur un run isolé |

Le nombre de coups doit être défini sans ambiguïté : préciser si une action refusée compte comme un coup, si les actions masquées sont exclues et comment sont traités les épisodes non résolus.

### Conditions de comparaison

- mêmes grilles de test ;
- même horizon ;
- mêmes règles de terminaison ;
- mêmes conditions d'évaluation ;
- plusieurs graines ;
- résultats séparés par niveau de difficulté.

Ne pas comparer un résultat sur un lot de 64 grilles à un résultat sur un lot de 256 sans tenir compte de la composition des lots et de la variance.

---

## 9. Tests techniques à écrire avant l'entraînement

La nouvelle politique factorisée introduit des cas limites. Ajouter des tests unitaires pour vérifier :

1. Une case remplie ne peut jamais être sélectionnée.
2. Une case sans chiffre légal ne peut jamais être sélectionnée.
3. Le masque chiffre correspond exactement aux règles ligne/colonne/bloc.
4. Le chiffre choisi est toujours légal localement.
5. Une grille terminée ne déclenche pas d'échantillonnage invalide.
6. Une grille sans action possible est traitée explicitement.
7. Les probabilités de la tête case sont normalisées sur les cases autorisées.
8. Les probabilités de la tête chiffre sont normalisées sur les chiffres autorisés pour la case sélectionnée.
9. La log-probabilité totale est la somme des deux log-probabilités.
10. Les gradients passent bien par les deux têtes.
11. Le calcul de la loss ignore proprement les épisodes terminés ou les actions invalides selon le contrat de l'environnement.
12. Les graines rendent l'échantillonnage reproductible.

Ne pas modifier le comportement de l'environnement existant sans tests de non-régression.

---

## 10. Découpage concret du travail pour l'IA qui reprend le projet

L'IA qui reprend le dépôt doit avancer dans cet ordre et ne pas passer à l'étape suivante si la précédente n'est pas validée.

### Phase 0 — Audit sans modification

- Lire `reinforce.py`, `reseau.py`, `oracle.py`, `sudoku_env.py`, `env_lots.py`.
- Lire les fiches de l'environnement, du réseau et des mesures.
- Identifier le contrat exact de `masque`, `etats`, `jouer_lot`, `evaluer_lot` et de la récompense.
- Vérifier le calcul des retours, de la baseline et l'utilisation réelle de la tête de valeur.
- Identifier les commandes exactes permettant de reproduire le run de référence.
- Rendre un compte rendu des constats et des éventuelles incohérences.
- **Ne modifier aucun fichier pendant cette phase.**

### Phase 1 — Référence reproductible

- Exécuter l'agent actuel sur un jeu de test fixe.
- Enregistrer configuration, commit, graines, métriques et logs.
- Vérifier que les résultats peuvent être reproduits.
- Ne pas modifier le réseau ni les récompenses.

### Phase 2 — Implémentation de la politique factorisée

- Ajouter une nouvelle architecture ou un mode distinct, sans écraser le réseau de référence.
- Implémenter la tête case (81 sorties).
- Implémenter la tête chiffre conditionnée par la case (9 sorties).
- Réutiliser les règles existantes pour les masques légaux.
- Ne pas ajouter de canaux, de mémoire des refus ou de labels.
- Adapter l'échantillonnage et le calcul de log-probabilité.
- Ajouter les tests unitaires listés plus haut.

### Phase 3 — Entraînement comparatif

- Utiliser le même REINFORCE et les mêmes récompenses que la référence.
- Lancer plusieurs graines.
- Comparer politique plate et politique factorisée sur les mêmes grilles.
- Publier les résultats, y compris les échecs et la variance.
- Ne pas ajouter le curriculum tant que cette comparaison n'est pas comprise.

### Phase 4 — Curriculum optionnel

- Ajouter une progression de difficulté uniquement dans le générateur d'épisodes.
- Ne pas modifier les observations, les labels (il n'y en a pas) ni la politique.
- Comparer avec le même protocole de test fixe.
- Garder cette fonctionnalité si le gain est robuste.

### Phase 5 — Décision sur l'algorithme RL

Seulement après les phases précédentes :
- décider si la tête de valeur doit être entraînée ;
- tester éventuellement un actor-critic ;
- comparer avec REINFORCE à données et architecture identiques ;
- ne pas introduire plusieurs changements simultanés.

---

## 11. Ce qui est explicitement hors périmètre

Pour éviter que l'implémentation dérive vers un système plus complexe que le besoin initial, ne pas faire dans cette série d'expériences :

- pas de 16-A : pas de mémoire des actions refusées ;
- pas de neuf canaux supplémentaires ;
- pas de labels du chiffre solution ;
- pas de labels de candidats générés par un solveur ;
- pas d'imitation d'un solveur ;
- pas de cible MRV ;
- pas de bonus de récompense heuristique ;
- pas de réseau plus gros par défaut ;
- pas de remplacement immédiat du CNN par un Transformer ou un GNN ;
- pas de changement simultané de l'architecture et de l'algorithme RL ;
- pas d'utilisation de la solution dans l'observation de l'agent à l'évaluation.

## 12. Résultat attendu

À la fin de cette démarche, on doit pouvoir répondre objectivement à ces questions :

1. La politique factorisée apprend-elle mieux que la politique plate à 729 actions ?
2. Le réseau apprend-il à choisir des cases et des chiffres utiles sans labels supplémentaires ?
3. Le masque fondé sur les règles suffit-il à maintenir la légalité des actions ?
4. Un curriculum de difficulté améliore-t-il la généralisation ?
5. Le problème restant vient-il de la représentation, du signal de récompense ou de REINFORCE ?

**Critère de réussite :** obtenir un gain reproductible sur des grilles de test jamais vues, sous un horizon fixé, sans fournir au modèle de labels de solution ou de stratégie. Le modèle doit apprendre à partir de l'état Sudoku, des règles légales et de l'expérience acquise dans l'environnement.

La priorité est donc : **audit → référence → politique factorisée → comparaison RL → curriculum optionnel**. 16-A est abandonnée pour cette trajectoire.
