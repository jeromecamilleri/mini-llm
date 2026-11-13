# Analyse du dépôt mini-llm : pistes pour l’apprentissage du Sudoku par renforcement

> Reconstitution de l’analyse initialement rédigée le 13 novembre 2026 à 14:01 (heure de Paris).  
> Dépôt étudié : [jeromecamilleri/mini-llm](https://github.com/jeromecamilleri/mini-llm)  
> Périmètre principal : `src/07_sudoku_rl/`, ses fiches, ses scripts et ses journaux d’expériences.

> **Note de méthode importante** — Ce document reprend l’analyse longue demandée, et non le résumé ultérieur centré sur l’expérience 16-A. Le texte original intégral n’étant pas récupérable mot pour mot dans l’historique disponible, cette version en reconstitue le contenu et le raisonnement. Les chiffres de performances issus de runs différents sont conservés comme indications historiques, mais signalés comme non directement comparables tant que leurs protocoles et logs ne sont pas rapprochés.

---

## 1. Diagnostic principal

J'ai parcouru ton dépôt et tes journaux d'expériences, notamment toute la partie `src/07_sudoku_rl`. J'ai également trouvé plusieurs travaux très proches, dont un résultat publié en octobre 2026 qui apporte une piste particulièrement intéressante.

**Mon diagnostic principal : tu n'es pas bloqué par la puissance de ta GTX 1070, ni principalement par la taille de ton réseau. Tu es bloqué par la formulation du problème d'apprentissage et par la représentation de l'action.**

Et il y a une piste que je testerais avant de changer de jeu ou d'architecture.

Ton environnement est déjà suffisamment élaboré pour poser une vraie question scientifique : est-ce qu'un réseau peut apprendre à résoudre progressivement un Sudoku à partir d'un signal d'oracle, et quelle représentation rend cet apprentissage accessible ?

Le problème actuel est que le réseau doit apprendre en même temps :
- à repérer quelles cases sont intéressantes ;
- à choisir un chiffre ;
- à comprendre les contraintes entre cases ;
- à éviter les propositions déjà refusées ;
- à organiser une suite de décisions qui aboutit à une grille complète.

Ces difficultés sont imbriquées dans une seule politique qui choisit une action parmi 729 possibilités. Le signal de récompense est relativement pauvre par rapport à la quantité d'information que ton environnement possède.

Je ne commencerais donc ni par augmenter le nombre de paramètres, ni par remplacer le CNN, ni par changer d'algorithme RL. Je commencerais par modifier la manière dont le problème est présenté à l'agent.

## 2. Ce que montrent les expériences : attention à l'interprétation des chiffres

Les mesures et les journaux montrent plusieurs comportements intéressants. Dans les résultats évoqués dans l'analyse initiale, on retrouve notamment les valeurs suivantes :

| Variante / résultat rapporté | Nombre de coups moyen |
|---|---:|
| Politique aléatoire avec masque | 92,8 |
| REINFORCE | 78,6 |
| Variante avec mémoire / refus | 63,6 |
| Référence idéale théorique ou heuristique | environ 51 |

Ces nombres donnent une indication de progression, mais **ils ne doivent pas être considérés comme une comparaison scientifique définitive** tant que les journaux exacts, les jeux de grilles, les niveaux de difficulté, les horizons et les conditions d'arrêt ne sont pas strictement identiques.

Les fiches de mesures du dépôt (`src/07_sudoku_rl/fiches/03_mesures.md`) mentionnent aussi des références par niveau : hasard avec masque autour de 87 coups sur facile et 94 sur diabolique ; agent qui choisit une case contrainte autour de 52 à 59 coups ; référence parfaite autour de 51 à 53 selon le niveau. Ces chiffres ne sont pas nécessairement contradictoires avec le tableau précédent : ils peuvent provenir de lots ou protocoles différents. Il ne faut pas les fusionner.

### Le problème des répétitions

Un diagnostic des journaux a également fait ressortir l'idée qu'une part importante des mauvais coups — rapportée dans l'analyse initiale à environ **60 %** — correspondrait à des répétitions de propositions déjà refusées.

C'est un point important. Si l'environnement ne change pas après une mauvaise proposition, l'agent peut revenir sur la même action ou sur une action équivalente. Il dépense alors des décisions à réapprendre une information que le système connaît déjà.

Mais il faut distinguer deux choses :
- empêcher mécaniquement de répéter une action rejetée ;
- apprendre à choisir une meilleure action.

La première peut améliorer le nombre de coups sans que le réseau ait réellement appris à résoudre le Sudoku. C'est pourquoi les répétitions doivent être mesurées séparément des placements corrects et du taux de résolution.

### Conséquence

Le taux de résolution seul est une métrique trompeuse. Avec un masque des coups légaux, un oracle qui valide les propositions et suffisamment de tentatives, un agent peut finir par remplir une grille sans avoir acquis une stratégie intéressante.

Les métriques prioritaires devraient être :
- nombre de coups jusqu'à résolution ;
- taux de propositions correctes ;
- nombre de propositions refusées ;
- taux de répétition des refus ;
- taux de résolution sous un horizon strict ;
- récompense cumulée ;
- généralisation à des grilles et difficultés jamais vues.

## 3. Le problème central : 729 actions, alors que deux décisions sont mélangées

Aujourd'hui, une action est un couple :

\[
a=(c,d)
\]

avec :
- \(c\) : une case parmi 81 ;
- \(d\) : un chiffre parmi 9.

L'espace d'action est donc :

\[
81 \times 9 = 729
\]

Le réseau choisit directement une des 729 possibilités.

C'est une représentation simple à coder, mais elle mélange deux décisions de nature différente :

1. **Quelle case faut-il traiter ?**
2. **Quel chiffre faut-il mettre dans cette case ?**

Or le Sudoku offre une structure naturelle pour séparer ces deux problèmes. Le choix d'une case peut s'appuyer sur son degré de contrainte : nombre de candidats restants, voisinage, appartenance à une ligne, colonne ou région. Le choix du chiffre est ensuite conditionné par la case sélectionnée.

Une politique factorisée pourrait s'écrire :

\[
P(c,d\mid s)=P(c\mid s)\,P(d\mid c,s)
\]

où \(s\) est l'état de la grille.

Le réseau aurait alors :
- une tête qui attribue un score aux 81 cases ;
- une tête qui attribue une probabilité aux 9 chiffres, conditionnée par la case choisie.

Cela ne réduit pas nécessairement le nombre total de sorties internes de façon spectaculaire, mais cela introduit une structure inductive cohérente avec le problème. Le réseau n'a plus à apprendre une distribution plate sur 729 couples sans expliciter la relation entre la case et le chiffre.

### Pourquoi cette factorisation est intéressante

Elle permet de diagnostiquer séparément :
- est-ce que le modèle choisit une bonne case ?
- est-ce qu'il connaît les candidats de cette case ?
- est-ce qu'il choisit le bon chiffre parmi les candidats ?

Si le modèle choisit une mauvaise case mais sait prédire le chiffre solution quand on lui impose une case, le problème est dans la tête de sélection. Si le choix de case est bon mais que le chiffre est mauvais, le problème est ailleurs.

Avec une sortie unique de 729 actions, ces erreurs sont plus difficiles à isoler.

## 4. Le signal d'apprentissage : ton oracle est beaucoup plus riche que la récompense utilisée

C'est probablement le point le plus important.

Ton oracle connaît la solution de la grille. Il peut donc répondre à une question très précise : pour cette case, est-ce que ce chiffre est celui de la solution ?

Le système dispose ainsi d'une information supervisée très forte. Pourtant, l'agent la reçoit principalement sous forme de récompense après avoir proposé un coup.

Cela revient à demander au réseau de découvrir par essais et erreurs une information que le générateur de données connaît déjà.

### Ce que l'oracle peut fournir

Pour une case vide donnée :
- les chiffres localement légaux, calculés avec les contraintes de ligne, colonne et bloc ;
- le chiffre effectivement présent dans la solution ;
- éventuellement un label de correction de la proposition.

Il faut cependant bien distinguer les **candidats locaux** du **chiffre solution**. Un candidat local est un chiffre qui ne crée pas de conflit immédiat. Il n'est pas nécessairement compatible avec une solution complète. Le chiffre solution, lui, est connu grâce à l'oracle.

### Proposition : apprentissage supervisé des candidats

Avant le RL, entraîner le réseau sur des exemples construits à partir de grilles résolues :

Entrée :
- grille partiellement remplie ;
- case ciblée ;
- éventuellement masque des candidats légaux.

Cibles :
- masque des chiffres localement candidats ;
- chiffre de la solution pour la case ciblée.

On peut entraîner deux sorties distinctes :
1. une prédiction multi-label des candidats ;
2. une classification du chiffre solution.

Cette étape permet de vérifier si le CNN est capable de comprendre les contraintes locales et d'exploiter l'information de la grille.

Ensuite seulement, on peut lui demander de choisir la case et d'apprendre une stratégie séquentielle.

### Pourquoi ce n'est pas de la triche

L'objectif est de construire un agent qui apprend à partir d'exemples produits par un environnement simulé. Utiliser la solution pour construire les labels d'entraînement est une forme de supervision classique.

Il faudra simplement veiller à ne pas fournir à l'agent, au moment de l'évaluation, des informations qui ne seraient pas disponibles dans le problème réel. La solution sert à construire les cibles d'apprentissage, pas à alimenter directement l'observation de l'agent au test.

## 5. Architecture actuelle : ce que je garderais et ce que je questionnerais

D'après les éléments examinés :
- entrée sous forme de canaux one-hot de la grille (10 canaux, ou 19 avec les canaux de refus) ;
- CNN résiduel de 4 blocs ;
- 64 canaux ;
- tête politique de 729 sorties ;
- tête de valeur scalaire ;
- environ 301 898 paramètres selon la fiche du réseau ;
- dernière couche politique initialisée à zéro, ce qui donne initialement une politique uniforme avant masquage.

C'est un réseau modeste. Pour une GTX 1070, ce n'est pas une architecture démesurée. Le nombre de paramètres ne semble pas être le premier verrou.

### La tête de valeur

Le réseau possède une tête de valeur, mais le code REINFORCE examiné semble ne pas l'utiliser dans la loss. La baseline est calculée à partir des retours observés à un même instant / tour, plutôt qu'à partir de la valeur prédite par le réseau.

Il y a donc un décalage entre l'architecture annoncée et l'algorithme réellement entraîné : la tête de valeur ne joue pas le rôle d'un critique actor-critic si elle n'est pas optimisée et utilisée pour calculer l'avantage.

Deux options cohérentes :
- retirer provisoirement la tête de valeur pour simplifier le modèle ;
- ou l'entraîner avec une loss de valeur et utiliser sa prédiction comme baseline dans un vrai actor-critic.

Je ne passerais pas directement à PPO sans avoir clarifié cette question et établi des baselines solides. Un algorithme plus complexe ne compensera pas une représentation de l'action ou un signal d'apprentissage mal adaptés.

### Le CNN et les contraintes globales

Le CNN est capable de propager de l'information à travers plusieurs couches, mais le Sudoku a une structure relationnelle très particulière : toutes les cases d'une même ligne, colonne ou région sont liées.

Un champ réceptif théorique couvrant toute la grille ne garantit pas que le réseau apprenne spontanément les règles de propagation de contraintes. À terme, on pourrait envisager :
- une architecture avec attention ;
- un graphe de contraintes ;
- des mécanismes explicites de propagation ;
- une architecture hybride CNN + contraintes.

Mais je ne commencerais pas par là. Il faut d'abord savoir si le réseau actuel peut apprendre correctement les candidats et la sélection de case.

## 6. Travaux proches : les pistes qui méritent d'être étudiées

### 6.1 Avra98 — sudoku-gpt2-curriculum

Référence : [Avra98/sudoku-gpt2-curriculum](https://huggingface.co/Avra98/sudoku-gpt2-curriculum)

La piste intéressante rapportée dans cette approche est de ne pas demander directement au modèle de choisir un coup parmi un espace d'actions global. Le modèle apprend plutôt à prédire les candidats d'une case, avec une progression de difficulté de type curriculum.

L'idée générale :
- commencer par des situations simples où les candidats sont faciles à déterminer ;
- augmenter progressivement la difficulté ;
- apprendre des contraintes ou candidats plus complexes ;
- laisser le modèle apprendre une représentation de la logique du Sudoku.

C'est très proche de la piste que je recommande pour ton dépôt : **apprendre d'abord ce qu'une case peut contenir, avant d'apprendre à choisir la prochaine action**.

Le modèle et le contexte d'entraînement de cette référence ne sont pas nécessairement comparables à ton CNN. Ce sont surtout les choix de formulation et de curriculum qui sont intéressants.

### 6.2 Avra98 — sudoku-latent-backtracking

Référence : [Avra98/sudoku-latent-backtracking](https://huggingface.co/Avra98/sudoku-latent-backtracking)

Cette piste s'intéresse à la capacité du modèle à gérer implicitement des raisonnements de type retour arrière (*backtracking*).

Le point à retenir n'est pas qu'il faudrait immédiatement ajouter un mécanisme de backtracking latent à ton projet. C'est plutôt que résoudre un Sudoku peut nécessiter de gérer des hypothèses, des contradictions et des retours sur des décisions antérieures.

Ton environnement actuel, qui pénalise une mauvaise proposition sans nécessairement faire évoluer la grille, ne représente pas forcément ce processus de recherche de la même manière qu'un solveur classique.

### 6.3 AdaC-GRPO — curriculum adaptatif pour Sudoku

Référence rapportée : [AdaC-GRPO, arXiv:2610.13184](https://arxiv.org/abs/2610.13184)

Le travail est présenté comme une approche de curriculum adaptatif pour des puzzles Sudoku à horizon long, avec une difficulté ajustée progressivement. Il est associé à une publication d'octobre 2026 et à une approche GRPO.

Les chiffres rapportés dans l'analyse initiale étaient :

| Variante évaluée dans le travail | Résultat rapporté |
|---|---:|
| Curriculum grossier | 0,405 |
| Curriculum fin adaptatif | 0,586 |
| Récompense partielle | 0,237 |
| Récompense binaire | 0,586 |

Ces valeurs doivent être lues dans le contexte de l'article et de son protocole : elles ne sont pas comparables aux nombres de coups de ton dépôt.

L'intérêt pour ton projet est surtout méthodologique :
- faire progresser la difficulté plutôt que présenter immédiatement des problèmes trop difficiles ;
- adapter le curriculum à la progression réelle du modèle ;
- comparer la récompense binaire à des formes de récompense plus informatives ;
- traiter le Sudoku comme une tâche à horizon long.

Attention : les modèles multimodaux ou modèles de langage de grande taille utilisés dans ce type de travail peuvent être très éloignés de ton réseau de quelques centaines de milliers de paramètres. Il faut transposer les idées, pas chercher à reproduire leurs moyens matériels.

## 7. Une architecture hybride possible : CNN + Candidate Prediction + RL

La direction que je privilégierais est une architecture en trois étapes.

### Étape 1 — Candidate Prediction

Le réseau reçoit la grille et apprend à prédire les candidats pour les cases vides.

On peut entraîner :
- une sortie par case et par chiffre pour le masque des candidats ;
- une sortie qui identifie le chiffre solution d'une case ciblée.

Cela fournit un objectif supervisé dense, beaucoup plus informatif qu'une récompense uniquement après un coup.

### Étape 2 — Choix de la case

Une tête séparée choisit la case à traiter. On peut initialiser ou comparer cette tête à une heuristique MRV (*Minimum Remaining Values*) : choisir une case qui possède le plus petit nombre de candidats.

MRV est une référence pertinente parce qu'elle formalise une stratégie classique de résolution. Elle ne résout pas à elle seule tous les cas, mais fournit un point de comparaison solide.

### Étape 3 — RL pour optimiser la séquence

Une fois le réseau capable de prédire les candidats et de choisir une case raisonnable, le RL peut apprendre à améliorer l'ordre des décisions et le comportement à long terme.

Le RL devient alors un outil d'optimisation de stratégie, et non le seul mécanisme chargé de découvrir simultanément les règles locales et la séquence complète.

## 8. Plan d'expériences proposé : 16-A, 16-B et 16-C

Ces expériences doivent rester séparées. Elles testent des hypothèses différentes et ne doivent pas être fusionnées dans un changement massif.

### Expérience 16-A — mémoire des refus et masque des actions déjà rejetées

**Statut : optionnelle.**

Hypothèse : une partie des coups perdus vient de la répétition de propositions déjà rejetées. Puisque l'état de la grille ne change pas après un refus, l'environnement peut mémoriser les actions refusées et les retirer du masque des actions disponibles.

À vérifier :
- la mémoire est bien réinitialisée lorsqu'un nouvel épisode commence ;
- une action refusée n'est masquée que dans l'état auquel ce refus se rapporte ;
- le masque ne rend pas artificiellement impossible une action dans un état ultérieur où elle pourrait être pertinente ;
- le modèle ne bénéficie pas d'une comparaison biaisée par un horizon ou un lot différent.

Mesures :
- nombre de répétitions ;
- nombre total de coups ;
- taux de propositions correctes ;
- taux de résolution sous horizon fixé ;
- récompense cumulée.

Cette expérience peut réduire les coups inutiles, mais ne prouve pas à elle seule que le réseau a appris à résoudre le Sudoku. Elle est utile comme optimisation d'environnement, mais elle ne doit pas être un prérequis de la piste principale.

### Expérience 16-B — factorisation de la politique : choix de case puis choix de chiffre

Hypothèse : le choix d'une case et le choix d'un chiffre sont deux sous-problèmes distincts. Une politique factorisée devrait rendre l'apprentissage plus structuré et les erreurs plus faciles à analyser.

Comparer :
- politique actuelle sur 729 actions ;
- tête de sélection sur 81 cases, puis tête de chiffre sur 9 valeurs ;
- éventuellement une sélection de case contrainte par les cases réellement disponibles.

Mesures :
- précision du choix de case ;
- précision du chiffre conditionnée à une case ;
- nombre de coups jusqu'à résolution ;
- taux de réussite sur les mêmes grilles.

C'est une piste structurelle importante, mais elle demande une modification du réseau et de la logique d'échantillonnage. Je la testerais après avoir construit des baselines fiables.

### Expérience 16-C — pré-entraînement supervisé à partir de l'oracle

Hypothèse : l'oracle donne une information qui peut être convertie en labels supervisés. Un réseau pré-entraîné sur les candidats et le chiffre solution devrait démarrer le RL avec une représentation plus utile.

Protocole :
1. générer un dataset de grilles partielles et de solutions ;
2. calculer les candidats locaux de chaque case vide ;
3. associer le chiffre solution comme cible distincte ;
4. entraîner le CNN sur ces tâches ;
5. évaluer sur des grilles disjointes ;
6. comparer un RL entraîné depuis zéro avec un RL initialisé par le modèle supervisé.

Mesures :
- précision/rappel des candidats ;
- précision du chiffre solution ;
- qualité de calibration ;
- nombre de coups et taux de résolution après RL ;
- généralisation par niveau de difficulté.

C'est la piste que je privilégierais en premier, car elle exploite directement une information déjà disponible dans l'environnement sans exiger une refonte complète de l'architecture.

## 9. Ordre de priorité et conclusion

Mon ordre de travail recommandé :

1. **Fiabiliser les mesures** : comparer les variantes sur les mêmes grilles, le même horizon et plusieurs graines.
2. **16-C : pré-entraînement supervisé** des candidats et du chiffre solution. C'est le test le plus informatif et le plus directement exploitable.
3. **16-B : politique factorisée** pour séparer le choix de case et le choix du chiffre.
4. **16-A : mémoire des refus** comme optimisation optionnelle, évaluée séparément.
5. Revenir sur la tête de valeur et le choix d'un algorithme actor-critic lorsque la formulation et les baselines sont claires.
6. Envisager une architecture avec attention ou graphe de contraintes seulement si les résultats montrent que le CNN ne représente pas correctement les relations du Sudoku.

### Ma recommandation finale

Je ne changerais pas de jeu. Je ne remplacerais pas tout de suite le CNN. Je ne chercherais pas non plus à faire tourner un modèle plus gros sur la GTX 1070.

Je conserverais le réseau actuel comme témoin et je construirais une expérience séparée autour de :

**CNN + Candidate Prediction + RL**

Le point décisif est de faire apprendre au réseau une représentation des candidats à partir de l'oracle, puis de lui apprendre à choisir les cases et à organiser ses décisions. Le RL devrait intervenir après cette étape, pour optimiser la stratégie séquentielle.

Si le réseau n'arrive pas à prédire correctement les candidats sur des grilles jamais vues, il faudra investiguer les données, l'encodage ou l'architecture. S'il y arrive mais que la politique RL reste mauvaise, le problème sera plus précisément situé dans la sélection des actions, le signal de récompense ou l'algorithme d'apprentissage.

C'est une démarche beaucoup plus informative que de multiplier les essais de REINFORCE en changeant plusieurs paramètres à la fois.

---

*Analyse du dépôt `jeromecamilleri/mini-llm` et recherches bibliographiques — 13 novembre 2026, 14:01 (heure de Paris).*
