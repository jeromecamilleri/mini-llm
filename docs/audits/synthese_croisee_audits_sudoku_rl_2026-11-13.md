# Synthèse croisée des audits Sudoku RL --- Astra GPT-6 et Claude Sonnet 5.5

**Date : 13 novembre 2026**\
ChatGPT GPT-6

**Projet :** `mini-llm` --- agent Sudoku par apprentissage par
renforcement\
**Nature :** synthèse et recommandations après l'audit technique de
GPT-6 Astra et la contre-expertise de Claude Sonnet 5.5.

------------------------------------------------------------------------

## 1. Objet et conclusion générale

Deux analyses ont été réalisées : un audit initial par GPT-6 Astra, puis
une contre-expertise par Claude Sonnet 5.5, qui a vérifié plusieurs
affirmations directement dans le code et effectué des contrôles courts
sur CPU.

Cette synthèse extrait les conclusions utiles pour décider de la suite.
Elle ne remplace pas les deux rapports. Les chiffres ci-dessous sont
ceux rapportés par les agents ; ils devront être reproduits dans un
protocole versionné avant de devenir des résultats de référence
définitifs.

**Conclusion principale : la politique factorisée est une expérience
pertinente, mais son intérêt n'est pas démontré. Son avantage apparent
est fortement influencé par sa distribution initiale.** Il faut
distinguer le gain provenant de cette distribution de celui provenant de
l'apprentissage ou de la nouvelle paramétrisation.

## 2. Conclusions communes et constats importants

### 2.1 L'environnement actuel dépend de l'oracle

Dans `SudokuEnvLots.step`, un coup n'est accepté et inscrit dans la
grille que s'il est effectué dans une case vide, respecte les
contraintes locales et correspond à la solution connue de l'oracle. Un
chiffre localement légal mais faux reçoit une récompense négative et ne
modifie pas la grille.

Conséquences : - les erreurs ne se propagent pas dans la grille ; - le
réseau peut essayer plusieurs chiffres pour une même case ; - il reçoit
une information de correction immédiate de l'oracle ; - l'agent apprend
à exploiter cet environnement avec oracle, et non encore à résoudre de
façon autonome un Sudoku sans oracle.

Sonnet a évalué les réseaux sans oracle sur les 512 grilles de test.
Selon son rapport, aucun réseau de référence n'a résolu une grille
entière dans ce test. Les réseaux ont produit environ 23,8 à 24,5 cases
fausses par grille, contre 23,3 pour le hasard plat.

Cette mesure ne démontre pas que le réseau n'a appris absolument aucune
information sur les chiffres. Elle montre qu'il ne sait pas résoudre les
grilles sans oracle dans le protocole testé.

**Décision :** ne pas changer l'environnement pour la première
comparaison architecturale. Ajouter une évaluation sans oracle,
clairement séparée des métriques principales de l'entraînement avec
oracle.

### 2.2 REINFORCE et `GAMMA = 0`

Astra a correctement relevé que la configuration de référence utilise
`GAMMA = 0`. Les retours utilisés par REINFORCE sont donc les
récompenses immédiates, sans retour actualisé des décisions futures.

Sonnet nuance l'interprétation : dans l'environnement avec oracle, le
nombre de placements corrects restant à effectuer est fixe. Le choix de
l'ordre des cases influence néanmoins le nombre d'essais faux
nécessaires. L'objectif immédiat est donc cohérent avec une partie de
l'objectif global, sans être équivalent à une optimisation complète de
la stratégie à long terme.

Autres constats rapportés par Sonnet : - la baseline par numéro de tour
a un effet limité dans les mesures réalisées ; - le bonus de victoire
augmente l'écart-type utilisé pour normaliser les avantages ; - la
documentation et les valeurs par défaut de `06_reinforce.py` ne
reflètent pas exactement la configuration de référence.

**Décision :** conserver REINFORCE, `GAMMA = 0` et les récompenses
inchangés pendant la comparaison des politiques. Corriger séparément la
documentation et les valeurs par défaut afin d'éviter les exécutions non
reproductibles. Ne pas introduire simultanément un nouvel algorithme
d'apprentissage.

### 2.3 Tête de valeur inutilisée

Les deux audits confirment que la tête de valeur est présente dans le
réseau mais n'est pas utilisée par la perte actuelle de REINFORCE.
Sonnet rapporte qu'après un pas d'apprentissage, ses paramètres restent
inchangés et que leurs gradients valent `None`.

Elle n'améliore donc pas actuellement l'apprentissage. Sonnet recommande
néanmoins de la conserver à ce stade, notamment pour préserver la
compatibilité des checkpoints et préparer une éventuelle utilisation
ultérieure.

**Décision :** conserver la tête de valeur pendant les expériences
comparatives, mais documenter explicitement qu'elle n'est pas entraînée
et ne joue aucun rôle dans la réduction actuelle de la variance.

## 3. Le résultat déterminant : le biais initial de la politique factorisée

### 3.1 Comparaison des hasards

Résultats rapportés par Sonnet sur les parties avec oracle :

  Agent                    Coups moyens   Coups faux   Répétitions
  ---------------------- -------------- ------------ -------------
  Hasard plat                      91,4         38,8           9,1
  Hasard factorisé             **82,8**     **30,2**       **5,6**
  Réseau plat entraîné       80,0 ± 1,2         27,6          14,7
  MRV                              56,1          3,5           0,5

Les résultats du hasard factorisé sont obtenus sans apprentissage.

Le premier coup correct est également plus probable avec la distribution
factorisée :

  Politique initiale                             Probabilité du premier coup correct
  -------------------------------------------- -------------------------------------
  Uniforme parmi les couples légaux (plate)                                  29,25 %
  Case uniforme, puis chiffre légal uniforme                                 33,21 %

Le hasard factorisé est ainsi à seulement 2,8 coups du réseau plat
entraîné, selon les résultats rapportés.

### 3.2 Pourquoi les distributions diffèrent

Pour une case `c`, notons `k_c` le nombre de chiffres légalement
possibles dans cette case.

-   Dans la politique plate initialisée avec des logits nuls, chaque
    couple légal a la même probabilité. La masse totale d'une case est
    donc proportionnelle à `k_c`.
-   Dans la politique factorisée initialisée avec des logits nuls,
    chaque case disponible reçoit la même probabilité, puis les chiffres
    légaux de cette case sont équiprobables.

La politique plate favorise donc initialement les cases qui ont le plus
de chiffres légaux. La politique factorisée choisit les cases
uniformément.

**Conséquence : une comparaison directe entre le réseau plat actuel et
un réseau factorisé confondrait l'effet de l'architecture avec celui du
biais initial de sélection des cases.**

### 3.3 Précision sur la correction proposée

Sonnet relève une imprécision dans la proposition d'Astra : un simple
réglage des poids initiaux ne peut pas ajouter un terme dépendant de
`k_c`, puisque cette quantité varie avec l'état du Sudoku.

Il s'agit d'un terme architectural calculé à partir du masque courant,
et non d'une simple initialisation des poids. Cette correction doit être
considérée comme une variante de politique à part entière. Elle utilise
uniquement le masque légal courant ; elle ne nécessite ni solution, ni
candidat calculé par solveur, ni label expert.

## 4. Comparaison expérimentale recommandée : plan 2 × 2

Pour isoler l'effet de la factorisation, Sonnet propose quatre
configurations :

  -----------------------------------------------------------------------
  Configuration           Architecture            Distribution initiale
                                                  des cases
  ----------------------- ----------------------- -----------------------
  **A --- Plate de        Politique plate         Proportionnelle à `k_c`
  référence**             actuelle                

  **B --- Factorisée      Politique factorisée    Uniforme
  naturelle**                                     

  **C --- Factorisée      Politique factorisée    Proportionnelle à `k_c`
  corrigée**              avec terme dépendant de 
                          `k_c`                   

  **D --- Plate           Politique plate avec    Uniforme
  corrigée**              correction dépendant de 
                          `k_c`                   
  -----------------------------------------------------------------------

Comparaisons utiles : - **B contre D :** la factorisation apporte-t-elle
quelque chose lorsque les distributions initiales des cases sont
comparables ? - **A contre D :** quel est l'effet du changement de
distribution dans la politique plate ? - **A contre B :** quel est le
gain total apparent, qui mélange biais initial et paramétrisation ? -
**C contre A :** la factorisation peut-elle retrouver le comportement
initial de la politique plate ?

L'égalité des distributions initiales ne garantit pas à elle seule une
comparaison parfaite des dynamiques d'apprentissage. Les
paramétrisations et les gradients restent différents. Il faudra
interpréter les résultats comme une comparaison expérimentale contrôlée,
pas comme une preuve mathématique d'équivalence.

## 5. Protocole expérimental à retenir

Avant d'entraîner les quatre configurations :

1.  **Figer le protocole** : mêmes données, horizon, taille de lot,
    nombre de mises à jour, optimiseur et récompenses.
2.  **Vérifier les distributions initiales** : mesurer les probabilités
    et les performances des quatre politiques sans entraînement.
3.  **Séparer validation et test** : ne pas utiliser le test pour
    choisir les hyperparamètres ou les checkpoints.
4.  **Comparer les configurations avec des graines appariées** et
    conserver les résultats individuels.
5.  **Tester les hyperparamètres sur validation** : les paramètres
    actuels ont été ajustés avec la politique plate ; ne pas supposer
    qu'ils sont optimaux pour les autres variantes.
6.  **Sauvegarder des checkpoints intermédiaires**, et pas seulement le
    dernier.
7.  **Enregistrer les résultats par graine et par grille**, avec les
    configurations et empreintes des données.
8.  **Définir à l'avance les critères de décision**, notamment le nombre
    de graines et le gain minimal considéré comme utile.

Sonnet estime qu'avec quatre graines par configuration, la différence
minimale détectable pourrait être de l'ordre de 2,7 coups dans ses
hypothèses statistiques. Il s'agit d'une estimation rapportée par
Sonnet, pas d'une puissance statistique définitivement établie. Elle
devra être recalculée à partir des données réelles et du protocole
final.

### Métriques principales et diagnostics

Métriques principales : - coups faux par grille ; - taux de résolution
avec oracle ; - résultats par niveau de difficulté.

Diagnostics utiles : - taux de coups faux selon `k`, le nombre de
chiffres légaux ; - probabilité de choisir un chiffre unique lorsqu'il
en existe un ; - entropie jointe de la politique ; - répétitions, à
interpréter comme une mesure de concentration de la politique et non
comme une preuve isolée d'absence de mémoire ; - résolution sans oracle,
comme évaluation distincte ; - temps et coût de calcul.

Pour une grille résolue, le nombre total de coups est égal au nombre de
cases vides plus le nombre de coups faux. Les deux métriques sont donc
redondantes dans ce cas ; les coups faux par grille sont plus
directement interprétables.

## 6. Décisions et contraintes à préserver

À préserver : - aucun label de chiffre solution ; - aucun label de
candidats calculés par solveur ; - aucune supervision MRV ; - aucune
imitation de solveur ; - aucune mémoire des actions refusées ; - aucune
observation supplémentaire pour mémoriser les refus ; - aucune tête
auxiliaire de prédiction de solution ou de candidats ; - l'oracle peut
rester dans l'environnement pour calculer les récompenses, sans
transmettre la solution au réseau ; - le masque légal calculé à partir
de l'état courant et des règles locales reste autorisé.

L'évaluation sans oracle ne doit pas être confondue avec une
modification immédiate de l'environnement d'entraînement.

## 7. Recommandation pour la suite

**Ne pas coder immédiatement la politique factorisée.**

### Étape 1 --- Fiabiliser l'évaluation

Ajouter les métriques manquantes et rendre reproductibles les mesures
rapportées : - coups faux par grille ; - diagnostics selon `k` ; -
mesure de la sélection des chiffres uniques ; - évaluation sans oracle
; - sauvegardes intermédiaires ; - configuration et empreintes des
données associées aux checkpoints.

### Étape 2 --- Tester la politique plate corrigée (D)

C'est la modification la plus limitée pour tester l'effet de la
distribution uniforme des cases sans introduire une nouvelle
architecture. Elle permettra de vérifier si une part importante du gain
attendu peut être obtenue sans factorisation.

### Étape 3 --- Réaliser la comparaison 2 × 2

Comparer les quatre configurations dans un protocole contrôlé, avec
plusieurs graines et une validation séparée.

### Étape 4 --- Décider à partir des résultats

La factorisation ne devra être retenue que si elle apporte un gain
robuste par rapport à la politique plate corrigée, et pas seulement par
rapport au hasard plat ou à la politique plate de référence.

Sonnet propose un seuil de trois coups pour considérer la factorisation
utile. Ce seuil est une proposition de décision expérimentale, pas une
propriété théorique ; il reste à valider avant les essais.

## 8. Questions encore ouvertes

Les audits ne permettent pas encore de conclure sur : 1. le gain propre
de la factorisation une fois le biais initial neutralisé ; 2. la part du
gain du réseau plat provenant du choix de la case ou du choix du chiffre
; 3. l'effet des hyperparamètres sur chaque paramétrisation ; 4.
l'intérêt d'entraîner la tête de valeur dans une phase ultérieure ; 5.
l'effet d'une modification de `GAMMA`, à étudier séparément ; 6. la
possibilité d'obtenir une résolution autonome sans oracle avec une
évolution de l'environnement et de l'apprentissage.

Ces questions doivent rester distinctes. Il ne faut pas modifier
plusieurs dimensions simultanément au risque de rendre les résultats
ininterprétables.

## 9. Conclusion

Les deux audits convergent sur le fait que l'architecture factorisée est
techniquement réalisable et compatible avec REINFORCE, sans ajout de
supervision experte.

La contre-expertise révèle cependant que son avantage apparent est
fortement influencé par sa distribution initiale : le hasard factorisé
réalise déjà une grande partie du gain observé.

La prochaine étape rationnelle consiste donc à **isoler l'effet du biais
de sélection des cases avant d'investir dans une nouvelle
architecture**.

L'objectif reste de construire un agent qui apprend à partir de l'état
du Sudoku, des règles accessibles dans l'environnement et de son
expérience, sans lui fournir de connaissance experte sous forme de
labels.

> À distribution initiale contrôlée, à observations, règles, récompenses
> et budget d'entraînement comparables, la politique factorisée
> apprend-elle réellement une meilleure politique que la politique plate
> ?

Tant que cette question n'a pas reçu de réponse expérimentale
reproductible, la factorisation reste une hypothèse à tester, pas une
solution acquise.
