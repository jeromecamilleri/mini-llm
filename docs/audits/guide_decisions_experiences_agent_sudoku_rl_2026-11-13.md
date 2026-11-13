# Guide de décision et feuille de route expérimentale --- Agent Sudoku RL

**Projet :** `mini-llm`\
**Périmètre :** `src/07_sudoku_rl/`\
**Date :** 13 novembre 2026\
**Statut :** document de référence pour les prochaines étapes ---
décisions validées et travaux à réaliser\
**Origine :** synthèse des audits GPT-6 Astra et Claude Sonnet 5.5,
arbitrage ChatGPT GPT-6 et décision du porteur du projet.

------------------------------------------------------------------------

## 1. Décision directrice du projet

Le projet poursuit désormais **deux objectifs complémentaires, évalués
séparément** :

1.  **Apprendre à jouer dans l'environnement avec oracle**, qui reste
    l'environnement d'entraînement principal actuel.
2.  **Mesurer la capacité à résoudre un Sudoku sans oracle**, dans un
    environnement d'évaluation distinct où les erreurs ont de vraies
    conséquences sur la grille.

L'objectif à long terme est de construire un agent générique qui apprend
à partir de l'état du Sudoku, des règles accessibles dans
l'environnement et de son expérience, sans recevoir de supervision
experte sous forme de labels.

Il ne faut pas confondre ces deux objectifs. Une bonne performance dans
l'environnement avec oracle ne démontre pas une capacité à résoudre un
Sudoku de manière autonome.

**Décision validée par le porteur du projet : conserver les deux
objectifs avec deux évaluations distinctes.**

## 2. Principes et contraintes non négociables

Les évolutions futures doivent respecter les principes suivants.

### 2.1 Pas de supervision experte

Ne pas ajouter : - de labels indiquant les chiffres de la solution ; -
de labels de candidats calculés par un solveur ; - de labels de choix de
case (MRV ou autre stratégie experte) ; - d'imitation de solveur ; - de
traces de résolution expertes utilisées comme cibles d'apprentissage ; -
de tête auxiliaire qui prédit la solution ou les candidats.

L'oracle peut rester dans l'environnement d'entraînement avec oracle
pour calculer les récompenses. Sa solution ne doit jamais être transmise
au réseau sous forme d'observation, de cible ou de label.

### 2.2 Règles locales autorisées

Le masque des actions légalement possibles, calculé uniquement à partir
de l'état courant de la grille et des règles du Sudoku (case vide,
ligne, colonne, bloc), est autorisé.

Ce masque applique les règles du jeu ; il ne constitue pas une
supervision experte.

### 2.3 Pas de mémoire des refus

Ne pas réintroduire : - de mémoire des actions refusées ; - de canaux
d'observation supplémentaires pour mémoriser les refus ; - de mécanisme
qui rende au réseau l'historique des propositions rejetées.

Un historique interne peut éventuellement être conservé par
l'environnement pour produire des métriques, à condition qu'il ne soit
pas accessible au réseau et n'influence pas ses décisions.

### 2.4 Expériences interprétables

Ne pas modifier simultanément plusieurs dimensions majeures. En
particulier, ne pas changer dans une même expérience l'architecture,
l'algorithme d'apprentissage, les récompenses et les transitions de
l'environnement.

Toute expérience doit préciser : - son hypothèse ; - la seule variable
principale modifiée ; - les variables maintenues constantes ; - les
métriques ; - les graines et jeux de données ; - les critères de
décision ; - les fichiers et commandes utilisés.

------------------------------------------------------------------------

## 3. Les deux objectifs et leurs évaluations

### 3.1 Évaluation A --- Environnement avec oracle

**Rôle : entraînement principal actuel et comparaison des politiques.**

Dans l'environnement actuel : - le masque élimine les actions localement
illégales ; - l'oracle indique si une proposition légale correspond à la
solution ; - une proposition correcte modifie la grille ; - une
proposition incorrecte reçoit une pénalité et ne modifie pas la grille.

Le réseau apprend donc à sélectionner efficacement des actions dans un
environnement qui lui fournit un retour de correction par oracle. Il
n'apprend pas seul les règles locales, puisque le masque les applique
déjà.

Cette évaluation mesure l'efficacité de l'agent dans ce contrat précis :
**jeu avec validation immédiate par oracle**.

Métriques principales : - nombre de coups faux par grille ; - taux de
résolution dans l'horizon fixé ; - résultats par difficulté.

Diagnostics complémentaires : - coups faux selon `k`, nombre de chiffres
légalement possibles dans la case choisie ; - probabilité de choisir un
chiffre unique lorsqu'il existe une case qui en possède un ; -
distribution des cases sélectionnées selon `k` ; - entropie de la
politique ; - répétitions d'actions ; - temps et coût de calcul.

Les répétitions sont une mesure de comportement/concentration de la
politique. Elles ne prouvent pas, à elles seules, un défaut de mémoire.

### 3.2 Évaluation B --- Environnement sans oracle

**Rôle : mesurer l'autonomie réelle de l'agent. Cette évaluation ne
remplace pas l'entraînement A dans la première phase.**

L'agent reçoit uniquement : - la grille courante ; - les informations
dérivées des règles locales autorisées ; - le masque des actions
localement légales, si cette interface est conservée.

Il ne reçoit aucun retour indiquant si un chiffre légalement proposé est
le bon chiffre de la solution.

Dans cet environnement : - une action localement légale est réellement
inscrite dans la grille ; - si elle est fausse par rapport à la
solution, elle peut créer une contradiction ultérieure ; - l'agent ne
peut pas compter sur l'oracle pour annuler silencieusement ses erreurs.

L'évaluateur peut connaître la solution pour calculer les résultats
après l'épisode, mais cette information doit rester strictement hors de
l'observation, des récompenses intermédiaires et des décisions de
l'agent.

**Point de conception à spécifier avant le développement :** définir
exactement la gestion d'une contradiction, la fin d'épisode, l'horizon
maximal et la possibilité --- ou non --- de revenir sur une décision. Ne
pas ajouter implicitement de backtracking ou d'aide experte. Si le
modèle ne dispose pas d'action de retour arrière, une erreur inscrite
dans la grille doit avoir les conséquences définies par cet
environnement, et non être corrigée automatiquement.

Métriques principales : - pourcentage de grilles entièrement résolues
correctement ; - pourcentage de grilles conduisant à une contradiction
; - nombre de cases correctement remplies ; - nombre de cases remplies
avant échec ou timeout ; - résultats par difficulté ; - taux de réussite
sans aide extérieure.

Cette évaluation doit être distincte de l'évaluation A. Les résultats
des deux environnements ne doivent jamais être agrégés dans un score
unique.

### 3.3 Évaluer les mêmes checkpoints dans les deux environnements

À terme, il est souhaitable d'évaluer les mêmes checkpoints entraînés
avec oracle dans A et dans B.

Cela permettra de mesurer si les compétences acquises dans
l'environnement avec oracle se transfèrent à la résolution autonome.

Précautions : - les checkpoints doivent être identifiés et associés à
leur configuration exacte ; - l'évaluation B ne doit fournir aucun
feedback oracle à l'agent ; - l'évaluateur peut comparer la grille
finale à la solution uniquement après l'épisode ; - les résultats A et B
doivent rester dans des tableaux et rapports distincts.

Il s'agit d'une mesure de transfert, pas d'une preuve que l'entraînement
avec oracle apprend nécessairement à résoudre sans oracle.

------------------------------------------------------------------------

## 4. Ce que les audits ont établi et ce qui reste à démontrer

### 4.1 Constats issus des audits

Les rapports d'Astra et de Sonnet indiquent notamment :

-   le réseau actuel utilise une politique plate de 729 couples
    case/chiffre ;
-   le corps CNN partage déjà ses paramètres entre les cases : la
    politique actuelle n'est pas une couche dense indépendante de 729
    sorties ;
-   la tête de valeur est présente mais n'est pas utilisée par la perte
    REINFORCE actuelle ;
-   la configuration de référence utilise `GAMMA = 0` ;
-   les erreurs sont refusées par l'oracle et ne modifient pas la grille
    dans l'environnement actuel ;
-   les hasards initiaux rapportés sont sensiblement différents selon la
    paramétrisation.

Ces constats sont ceux des audits et doivent être revalidés dans le code
au moment de toute modification concernée.

### 4.2 Résultat déterminant sur le biais initial

Résultats de simulation rapportés par Sonnet pour l'environnement avec
oracle :

  Agent                    Coups moyens   Coups faux   Répétitions
  ---------------------- -------------- ------------ -------------
  Hasard plat                      91,4         38,8           9,1
  Hasard factorisé                 82,8         30,2           5,6
  Réseau plat entraîné       80,0 ± 1,2         27,6          14,7
  MRV                              56,1          3,5           0,5

Probabilité rapportée d'un premier coup correct : - hasard plat : 29,25
% ; - hasard factorisé : 33,21 %.

Ces résultats montrent qu'une comparaison directe entre le réseau plat
actuel et le réseau factorisé serait biaisée : le hasard factorisé part
déjà avec une distribution plus favorable dans cet environnement.

Les valeurs sont des résultats rapportés par les audits. Avant de les
utiliser comme références officielles, il faut les reproduire avec un
script versionné, des graines documentées et les mêmes conditions
d'évaluation.

### 4.3 Ce que ces résultats ne démontrent pas

Ils ne démontrent pas : - que la factorisation est meilleure après
apprentissage ; - que le réseau factorisé apprendra plus vite ; - que le
réseau actuel a appris une stratégie autonome de résolution ; - que
`GAMMA = 0` est optimal ; - que le choix de case explique à lui seul le
gain du réseau entraîné.

Ces questions nécessitent des expériences dédiées.

------------------------------------------------------------------------

## 5. Décision sur la politique factorisée : comparaison contrôlée 2 × 2

**Ne pas coder ni adopter la politique factorisée avant d'avoir isolé le
biais de distribution initiale.**

Le plan expérimental proposé comporte quatre variantes :

  -----------------------------------------------------------------------
  Variante                Politique               Distribution initiale
                                                  des cases
  ----------------------- ----------------------- -----------------------
  **A --- Plate de        Politique plate         Proportionnelle à `k_c`
  référence**             actuelle                

  **B --- Factorisée      Politique factorisée    Uniforme
  naturelle**                                     

  **C --- Factorisée      Politique factorisée    Proportionnelle à `k_c`
  corrigée**              avec correction         
                          dépendant de `k_c`      

  **D --- Plate           Politique plate avec    Uniforme
  corrigée**              correction dépendant de 
                          `k_c`                   
  -----------------------------------------------------------------------

`k_c` désigne le nombre de chiffres légalement possibles dans la case
`c`, calculé depuis le masque courant.

Comparaisons à interpréter : - **B contre D :** effet de la
factorisation lorsque la distribution initiale des cases est comparable
; - **A contre D :** effet du changement de distribution dans la
politique plate ; - **A contre B :** différence totale, qui mélange
distribution initiale et paramétrisation ; - **C contre A :**
comportement de la factorisation lorsque la distribution initiale vise à
retrouver celle de la politique plate.

Le terme dépendant de `k_c` est calculé dynamiquement depuis le masque.
Ce n'est pas une simple initialisation de poids. Il doit être documenté
comme une variante de politique.

Ce mécanisme respecte les contraintes sans labels s'il utilise
exclusivement le masque légal courant. Il ne doit jamais consulter la
solution ou les candidats d'un solveur.

**Critère de décision :** ne retenir la factorisation que si elle
apporte un gain reproductible par rapport à la politique plate corrigée
D, et pas seulement par rapport à A. Sonnet a proposé un seuil indicatif
de trois coups ; ce seuil doit être validé et fixé avant les essais, et
non choisi après observation des résultats.

------------------------------------------------------------------------

## 6. Protocole expérimental pour comparer les politiques

### 6.1 Conditions communes

Pour la première comparaison, conserver autant que possible : - le même
jeu de grilles d'entraînement ; - les mêmes ensembles de validation et
de test ; - les mêmes observations ; - le même masque légal ; - les
mêmes récompenses et transitions de l'environnement A ; - le même
horizon ; - la même taille de lot ; - le même optimiseur ; - le même
nombre de mises à jour, sauf expérience explicitement consacrée au
budget de calcul.

La référence rapportée par Astra/Sonnet mentionne notamment un
mini-pool, un lot de 64, un horizon de 150, 500 mises à jour, Adam,
`lr=3e-3` et `GAMMA=0`. Ces paramètres sont un point de départ à
confirmer dans le code et les scripts avant de les figer.

### 6.2 Initialisation et hasard

Avant tout entraînement : - mesurer les distributions initiales A, B, C
et D ; - vérifier les probabilités sur plusieurs grilles ; - comparer
les performances sans apprentissage ; - vérifier que les corrections
produisent bien la distribution annoncée ; - contrôler les masques vides
et les grilles terminées.

Ne pas conclure à un gain d'apprentissage à partir d'un gain déjà
présent avant entraînement.

### 6.3 Graines et statistiques

-   Utiliser des graines appariées entre variantes lorsque c'est
    possible.
-   Enregistrer les résultats de chaque graine, pas seulement la
    moyenne.
-   Distinguer la variabilité entre entraînements de la variabilité des
    tirages d'évaluation.
-   Ne pas considérer plusieurs tirages sur une même grille comme autant
    de grilles indépendantes.
-   Commencer avec plusieurs graines ; augmenter leur nombre si l'effet
    observé est faible ou incertain.
-   Fixer à l'avance le gain minimal utile et les critères d'arrêt.

Les estimations de puissance statistique avancées dans la
contre-expertise sont indicatives et doivent être recalculées à partir
du protocole réellement retenu.

### 6.4 Hyperparamètres et checkpoints

Les hyperparamètres actuels ont été ajustés avec la politique plate. Ne
pas supposer qu'ils sont optimaux pour les autres variantes.

-   Utiliser la validation pour choisir les hyperparamètres et
    checkpoints.
-   Garder le test final à l'écart de ces choix.
-   Sauvegarder des checkpoints intermédiaires.
-   Sauvegarder la configuration, l'état de l'optimiseur si nécessaire,
    la graine, la version du code et les empreintes des données.
-   Séparer les résultats par architecture/variante dans les scripts
    d'évaluation.

### 6.5 Mesures à enregistrer

Pour chaque variante, graine et grille : - coups faux ; - résolution ou
échec à l'horizon ; - difficulté ; - distribution des actions selon
`k_c` ; - taux de sélection d'une case avec chiffre unique lorsqu'il en
existe ; - entropie jointe ; - répétitions ; - temps et coût de calcul.

Pour une grille résolue, le nombre total de coups peut être redondant
avec le nombre de cases vides et les coups faux. Privilégier les
métriques directement interprétables et préciser leur définition.

------------------------------------------------------------------------

## 7. REINFORCE, tête de valeur et autres évolutions

### 7.1 REINFORCE

Pour isoler l'effet de la politique, conserver dans un premier temps : -
REINFORCE ; - `GAMMA = 0` ; - les récompenses actuelles ; - les autres
hyperparamètres de référence, après vérification.

Ne pas passer simultanément à actor-critic. Une éventuelle étude de
`GAMMA` devra faire l'objet d'une expérience séparée, après la
comparaison des politiques.

### 7.2 Tête de valeur

La tête de valeur actuelle est rapportée comme inutilisée dans la perte.
La conserver provisoirement évite de mélanger une modification de
structure et une comparaison de politiques, et préserve la compatibilité
des checkpoints.

La documenter explicitement comme non entraînée et sans effet actuel sur
la variance.

Son utilisation future dans une approche actor-critic ou comme baseline
apprise constitue une expérience séparée, qui ne doit pas être anticipée
dans la comparaison actuelle.

### 7.3 Documentation et reproductibilité

Vérifier et corriger, dans une tâche dédiée : - les docstrings qui ne
correspondent plus à la configuration réelle ; - les valeurs par défaut
des scripts qui diffèrent des expériences de référence ; - les
descriptions de la tête de valeur ; - la description de l'oracle et du
masque dans le README et les fiches.

Ne pas mélanger ces corrections documentaires avec une modification
algorithmique importante.

------------------------------------------------------------------------

## 8. Feuille de route recommandée

### Phase 0 --- Stabiliser et documenter la référence

-   Vérifier le code et les paramètres réellement utilisés.
-   Figer les jeux de données, graines et configurations de référence.
-   Vérifier les métriques et les scripts d'évaluation.
-   Clarifier dans la documentation le contrat de l'environnement avec
    oracle.

**Livrable :** référence reproductible de la politique plate actuelle.

### Phase 1 --- Compléter les évaluations

-   Ajouter les diagnostics par `k_c`.
-   Ajouter la mesure de sélection des chiffres uniques.
-   Ajouter les résultats détaillés par grille et par graine.
-   Préparer l'évaluation sans oracle sans modifier l'entraînement.
-   Définir précisément les transitions, contradictions et terminaisons
    de l'environnement B.

**Livrable :** deux évaluations documentées, A avec oracle et B sans
oracle.

### Phase 2 --- Tester la politique plate corrigée D

-   Implémenter la variante D seulement après spécification et
    validation mathématique.
-   Vérifier sa distribution initiale.
-   Mesurer son comportement sans apprentissage.
-   L'entraîner dans les mêmes conditions que la référence.
-   Comparer A et D.

**Livrable :** mesure de l'effet du choix uniforme des cases sans
nouvelle tête factorisée.

### Phase 3 --- Comparaison factorisée 2 × 2

-   Implémenter B et C si le protocole le justifie.
-   Vérifier les distributions initiales et les log-probabilités.
-   Réaliser la comparaison avec graines appariées.
-   Évaluer les checkpoints dans A et B, sans mélanger les scores.

**Livrable :** conclusion expérimentale sur l'intérêt propre de la
factorisation.

### Phase 4 --- Étudier le transfert vers la résolution autonome

-   Comparer les checkpoints entraînés avec oracle dans l'environnement
    sans oracle.
-   Identifier les erreurs et contradictions.
-   Ne modifier l'environnement ou l'algorithme qu'à partir d'une
    hypothèse explicite et d'une expérience isolée.

**Livrable :** mesure du transfert de l'apprentissage avec oracle vers
la résolution autonome.

Les phases ne doivent pas être interprétées comme une obligation de
complexifier le modèle. Si une expérience montre que la politique plate
corrigée suffit, il est parfaitement acceptable de ne pas retenir la
factorisation.

------------------------------------------------------------------------

## 9. Règles de travail pour GitHub Copilot

Tout agent IA intervenant sur ce projet doit respecter les règles
suivantes.

### Avant de coder

1.  Lire ce document et les rapports d'audit associés.
2.  Examiner le code actuel ; ne pas supposer que les rapports décrivent
    encore exactement la branche.
3.  Reformuler l'objectif de la tâche et la variable expérimentale
    modifiée.
4.  Lister les fichiers qui seraient modifiés.
5.  Signaler les ambiguïtés et les risques avant toute modification.

### Pendant le travail

-   Ne pas modifier des fichiers hors périmètre sans accord.
-   Ne pas lancer d'entraînement coûteux sans autorisation explicite.
-   Ne pas lancer de vérifications exploratoires longues sans nécessité
    ; si un contrôle important est nécessaire, le proposer avant de
    l'exécuter.
-   Ne jamais introduire de labels, de supervision experte ou de mémoire
    des refus.
-   Ne pas changer plusieurs dimensions expérimentales en même temps.
-   Ne pas présenter une hypothèse comme un fait établi.

### Après le travail

Fournir un compte rendu concis : - fichiers modifiés ; - changements
effectués ; - tests exécutés et résultats ; - commandes permettant de
reproduire ; - limites ou contrôles non réalisés ; - impact sur les deux
évaluations A et B.

Si la tâche est un audit ou une analyse, ne modifier aucun fichier et ne
lancer aucun entraînement. Produire un rapport limité aux conclusions
utiles, avec références précises au code.

------------------------------------------------------------------------

## 10. Décisions validées et questions ouvertes

### Décisions validées

-   Le projet poursuit deux objectifs : apprentissage avec oracle et
    résolution autonome sans oracle.
-   Ces objectifs disposent de deux évaluations distinctes.
-   L'environnement avec oracle reste l'environnement d'entraînement
    principal actuel.
-   L'évaluation sans oracle doit mesurer l'autonomie sans transmettre
    d'information de solution à l'agent.
-   Les contraintes sans labels, sans imitation et sans mémoire des
    refus sont maintenues.
-   La politique factorisée n'est pas adoptée avant une comparaison
    contrôlée.
-   Le biais initial de distribution doit être isolé.
-   Les évolutions majeures doivent être testées séparément.

### Questions encore ouvertes

-   Définition précise des contradictions et fins d'épisode dans
    l'environnement sans oracle.
-   Choix final de la correction mathématique des politiques C et D.
-   Nombre de graines et seuil de gain minimal utile.
-   Résultat réel de la comparaison 2 × 2.
-   Effet éventuel de `GAMMA > 0`.
-   Utilisation future de la tête de valeur.
-   Capacité de transfert des checkpoints vers la résolution autonome.

Aucune de ces questions ne doit être considérée comme déjà résolue.

------------------------------------------------------------------------

## 11. Synthèse finale

Le projet doit avancer par expériences contrôlées plutôt que par
accumulation de mécanismes.

La priorité est de disposer de deux évaluations fiables : - l'une qui
mesure l'efficacité de l'apprentissage dans l'environnement avec oracle
; - l'autre qui mesure la capacité réelle à résoudre sans oracle.

La comparaison de la politique plate et de la politique factorisée doit
ensuite isoler le biais initial de sélection des cases. La factorisation
ne sera retenue que si elle apporte un avantage reproductible au-delà de
ce biais.

**Question expérimentale centrale :**

> À distribution initiale contrôlée, à observations, règles, récompenses
> et budget d'entraînement comparables, la politique factorisée
> apprend-elle réellement une meilleure politique que la politique
> plate, et cet apprentissage se transfère-t-il à la résolution sans
> oracle ?

Ce document constitue le guide de décision pour les prochaines
évolutions. Toute proposition qui s'en écarte doit expliciter la raison,
les conséquences sur les objectifs et le protocole permettant de
l'évaluer.
