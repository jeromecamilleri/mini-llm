# Audit technique — Phase 0

> **Auteur : GPT-6 Astra (identifiant de modèle : `gpt-6-astra`)**, via GitHub Copilot CLI, le 13 novembre 2026.
> Une contre-expertise indépendante existe : `docs/contre_expertise_phase0_sudoku_rl_2026-11-13_sonnet.md`.

**La politique factorisée est une expérience pertinente, mais rien ne démontre encore que les 729 actions sont la cause du plateau.** Deux limites sont plus solidement établies : l’environnement protège le joueur grâce à la solution, et la référence actuelle apprend uniquement à partir de la récompense immédiate (`GAMMA = 0`).

**Aucun fichier modifié, aucun entraînement lancé.** J’ai lu les outils, scripts, fiches, plans et journaux concernés, vérifié les références externes et effectué quelques contrôles courts sur CPU.

Les chemins abrégés `tools/`, `scripts/` et `fiches/` désignent ceux de `src/07_sudoku_rl/`.

## 1. Architecture actuelle : faits vérifiés

| Élément | Fonctionnement réel | Source |
|---|---|---|
| Observation | Grille courante, encodée en `(B, 10, 9, 9)` : vide et chiffres 1–9 | `tools/reseau.py:18`, `tools/env_lots.py:125` |
| Corps du réseau | Convolution 3×3, 64 canaux, ReLU, puis 4 blocs résiduels de 2 convolutions 3×3 | `tools/reseau.py:43–68` |
| Politique | Convolution 1×1 donnant 9 scores par case, réordonnés en 729 scores | `ReseauSudoku.forward`, ligne 70 |
| Valeur | Moyenne spatiale des 64 caractéristiques, puis couche linéaire vers un scalaire | Même fonction |
| Taille | **301 898 paramètres**, dont 65 pour la tête de valeur | Vérifié par instanciation |
| Initialisation | Poids et biais de la tête politique nuls : distribution uniforme sur les actions autorisées | `ReseauSudoku.__init__` |
| Action | `ligne × 81 + colonne × 9 + chiffre − 1` | `tools/sudoku_utils.py:22–38` |
| Choix pendant l’entraînement | Tirage selon le softmax masqué, pas `argmax` | `tools/reinforce.py:33`, `jouer_lot` |

Le champ réceptif théorique du CNN couvre toute la grille. **Cela rend l’information accessible, sans garantir que le réseau sache exploiter les relations ligne/colonne/bloc.** Il n’a ni invariance explicite aux permutations des chiffres, ni mécanisme dédié de raisonnement Sudoku.

### Respect des contraintes

Pour les quatre checkpoints de référence, j’ai vérifié une entrée à **10 canaux**, sans refus visibles. Le chemin d’entraînement correspondant n’emploie ni labels solutions, ni supervision de candidats, ni imitation MRV.

Deux réserves :

- Les variantes historiques `--refus` et `memoire=True` existent encore. `06e_reference.py` accepte automatiquement un checkpoint à 19 canaux : **le protocole ne verrouille pas techniquement l’exclusion de ces variantes**.
- `SudokuEnvLots` conserve toujours `self.refuse`, même sans mémoire activée. Dans la référence, ce tableau sert au comptage des répétitions, **pas à la décision**. Il n’y a donc pas de mémoire accessible au joueur, mais il existe bien un historique interne de mesure.

Sources : `tools/env_lots.py:60–127`, `tools/reseau.py:87`, `tools/protocole.py:102–127`.

## 2. Environnement : ce que le modèle apprend réellement

### Observations, actions et masques

Les trois implémentations principales du masque sont cohérentes :

- `evaluation.masque_actions` ;
- `reseau.masques_actions` ;
- `SudokuEnvLots.masques`.

Elles autorisent un chiffre si la case est vide et si ce chiffre est absent de sa ligne, de sa colonne et de son bloc. **Aucune consultation de la solution dans ces masques.**

Contrôles effectués : égalité des trois masques sur les **512 grilles de référence**, et bijection correcte des 729 encodages d’action.

Attention au nom `SudokuEnv.actions_legales()` : cette méthode exclut seulement les cases remplies, **pas les doublons**. Elle n’est pas le masque utilisé par REINFORCE. Son nom peut induire en erreur.

Source : `tools/sudoku_env.py:199`.

### Récompenses et transitions

Dans `Oracle.verifier` et `SudokuEnvLots.step` :

| Action | État suivant | Récompense |
|---|---|---:|
| Chiffre correspondant à la solution, placement autorisé | Chiffre posé | +1 |
| Mauvais chiffre, même localement légal | Grille inchangée | −1 |
| Dernier placement correct | Grille terminée | +11 |
| Dernière tentative au timeout, sans victoire | Placement éventuellement effectué | **0, remplace le +1 ou −1** |

Sources : `tools/oracle.py`, `tools/sudoku_env.py:82`, `tools/env_lots.py:85`.

Les environnements scalaire et par lots ont donné les mêmes grilles, récompenses, terminaisons et masques sur un contrôle court de **554 actions**, couvrant victoire et timeouts.

### Limite majeure : l’oracle ne sert pas seulement à noter

**Fait : la solution décide aussi si l’action modifie la grille.**

Le modèle ne reçoit pas la bonne réponse comme cible supervisée, ce qui respecte cette interdiction. Mais le système lui évite toute branche erronée : un chiffre faux, pourtant légal localement, n’entre jamais dans la grille.

**Conséquence : nous mesurons un joueur avec validation immédiate par oracle, pas encore un solveur autonome utilisant seulement les règles locales.** La baseline MRV bénéficie exactement de la même protection.

Ce n’est pas une raison pour changer maintenant l’environnement : cela détruirait la comparaison. En revanche, la formulation « modèle prêt à jouer » doit préciser **« dans cet environnement avec oracle »**.

### Pourquoi les répétitions persistent

Après un refus, la grille reste identique. Pour une politique sans mémoire :

- les probabilités restent identiques ;
- le tirage peut répéter l’action ;
- un joueur déterministe répète nécessairement le même mauvais choix.

**La factorisation ne supprime pas cette propriété.** Elle peut améliorer les probabilités, pas rendre le réseau conscient du refus précédent.

De plus, le nombre de tentatives restantes n’est pas observé : la même grille peut être vue loin du timeout ou juste avant. La politique ne peut pas adapter son comportement à ce budget.

## 3. REINFORCE : calculs corrects et limites

### Fonctionnement vérifié

`tools/reinforce.py` réalise :

1. Une collecte de trajectoires sans gradient.
2. Le calcul rétrograde :
   \[
   G_t=r_t+\gamma G_{t+1}
   \]
3. Une moyenne des retours **par numéro de tour**, sur les parties encore actives.
4. Le centrage puis la normalisation des avantages.
5. La perte :
   \[
   L=-\frac{1}{N}\sum_{\text{actions actives}}
   A_t\log\pi_\theta(a_t\mid s_t)
   \]
6. Une accumulation des gradients par paquets, puis un seul pas d’Adam.

Sources : `retours`, ligne 57 ; `pas_reinforce`, ligne 67.

Le recalcul des décisions est cohérent avec la collecte pour la référence : mêmes observations, mêmes masques, aucun changement de poids entre collecte et recalcul, pas de dropout ni de normalisation dépendante du lot.

### La tête de valeur n’est pas entraînée

**Confirmé dans le code et par un calcul de gradient : ses poids et son biais ont `grad=None`.**

La sortie valeur est systématiquement ignorée dans la perte. La tête conserve donc ses paramètres initiaux ; sa sortie peut néanmoins varier parce que le corps partagé apprend.

**Ce n’est pas de l’actor-critic.** La fiche `04_reseau.md`, lorsqu’elle dit que la valeur réduit le bruit, décrit une utilisation envisagée, pas celle du REINFORCE actuel.

### `GAMMA = 0` change l’objectif

La référence utilise :

\[
G_t=r_t
\]

Un coup n’est donc pas récompensé pour avoir facilité les décisions suivantes. Il est évalué sur sa correction immédiate.

**Interprétation :** cela facilite le signal local, mais ne constitue pas une optimisation explicite de la stratégie à long terme. Une factorisation peut aider à sélectionner des coups immédiatement fiables ; elle ne rétablit pas le crédit temporel manquant.

Pour `gamma > 0`, le code utilise les retours actualisés sans pondération extérieure `gamma**t`. C’est une variante courante, mais il ne faut pas la présenter sans nuance comme l’estimateur exact de l’objectif actualisé depuis le début de l’épisode.

### Baseline et normalisation : réserves techniques

Dans `pas_reinforce` :

- La moyenne du tour inclut le retour de l’action elle-même. L’estimateur n’a donc pas exactement les propriétés d’une baseline indépendante de cette action.
- Avec une seule partie encore active, son avantage centré vaut **zéro** : cette fin de trajectoire n’apporte plus de signal.
- La normalisation par l’écart-type du même échantillon modifie aussi l’estimateur.
- La division par le nombre total d’actions produit une pondération par décision ; les épisodes longs occupent davantage de place.
- Avec une seule action active au total, `ret.std()` peut produire `NaN` malgré l’ajout de `1e-8`.

Ces points ne démontrent pas la cause du plateau. Ils empêchent toutefois de considérer l’algorithme comme un REINFORCE théorique sans particularités.

Enfin, **changer vers actor-critic tout en conservant `gamma=0` ne créerait pas d’apprentissage à long terme** : le critique améliorerait surtout la comparaison locale.

## 4. Problèmes et incohérences détectés

### Robustesse technique

| Constat | Impact | Source |
|---|---|---|
| Masque entièrement faux remplacé par 729 scores `−1e9` | Le softmax devient uniforme sur des actions toutes interdites, plutôt que « aucune action » | `reseau.py:70` |
| Grille déjà complète non terminée au `reset` | Elle peut consommer des coups jusqu’au timeout | `env_lots.py:60`, `sudoku_env.py:75` |
| Peu de validation des entrées de l’environnement par lots | La cohérence puzzle/solution est supposée ; la victoire y est déterminée par zéro case vide | `env_lots.py:52–115` |
| Tests surtout intégrés aux leçons | Plusieurs contrôles affichent un écart sans faire échouer le script | `scripts/04_reseau.py`, `05_env_lots.py` |

Le masque vide des parties déjà terminées ne pollue pas actuellement la perte, car leurs actions sont exclues. **Mais il faut traiter explicitement ce cas avant une politique à deux distributions.**

### Interprétations historiques à corriger

**Mes affirmations précédentes étaient trop catégoriques sur plusieurs points :**

- **« L’écart 81,7 contre 78,4 est attribuable à la généralisation » : non démontré.** Checkpoints, grilles et tirages diffèrent. Les CSV de l’ancien essai 10 et de la nouvelle graine 0 divergent dès l’itération 6 : ce n’est pas une reproduction numérique identique.
- **« Zéro erreur avec un seul candidat prouve que le réseau a appris » : faux.** Le masque impose le seul chiffre possible. Il faut mesurer si le réseau choisit davantage ces cases qu’une baseline.
- **« À partir d’un écart de logits de 20, plus aucun gradient » : trop fort.** Contrôle float32 : avec `[20, 0]`, le gradient du meilleur logit vaut zéro mais celui de l’autre vaut encore environ `2 × 10⁻⁹`. Une saturation bien plus forte peut effectivement tout annuler.
- **« Entropie saine entre 0,9 et 1,2 » : repère historique, pas critère universel.** Une politique parfaite déterministe aurait une entropie nulle. Une entropie élevée ne prouve pas non plus une bonne stratégie.
- **« Le hasard résout toujours » : seulement un constat empirique ou une propriété asymptotique sous certaines conditions.** Sous horizon fini, des échecs restent possibles. Le journal des baselines affiche déjà 99,8 % sur facile ; le total arrondi à 100,0 % masque cette nuance.

Sources : `docs/suivi_sudoku_rl.md`, `fiches/03_mesures.md`, `scripts/06c_diagnostic.py`, `06d_divergence.py`, journaux de référence.

## 5. Politique factorisée : pertinence et pièges

### Architecture minimale pertinente

Conserver le corps CNN et ses caractéristiques spatiales `h(case)` :

- **Tête case** : projection de chaque vecteur de 64 caractéristiques vers un score → 81 logits.
- **Tête chiffre** : sélectionner `h(case_choisie)`, puis le projeter vers 9 logits.

La tête chiffre est ainsi réellement conditionnée par la case et par l’état. Une tête donnant seulement neuf logits globaux indépendants de la case serait une restriction injustifiée.

Avec des projections linéaires, la tête chiffre ressemble beaucoup à la convolution 1×1 existante. Ajouter la tête case représente seulement **65 paramètres supplémentaires**, à corps et tête valeur conservés.

**Il ne s’agit donc pas de remplacer une énorme tête dense de 729 neurones : la tête actuelle partage déjà ses poids entre les cases.**

### Masques et log-probabilité

À partir du masque actuel `M(case, chiffre)` :

\[
M_{\text{case}}(c)=\bigvee_d M(c,d)
\]

Puis :

\[
p(c)=\operatorname{softmax}_{c\ \text{autorisé}}(u_c)
\]

\[
p(d\mid c)=\operatorname{softmax}_{d:\,M(c,d)}(v_{c,d})
\]

Pour une action jouée :

\[
\log p(c,d)=\log p(c)+\log p(d\mid c)
\]

**REINFORCE est directement compatible :** le même avantage multiplie cette somme. Aucun besoin de différencier le tirage discret de la case, aucun label ajouté.

Pendant le recalcul, il faut reprendre **la case réellement jouée**, pas en tirer une nouvelle.

### Expressivité : distinction importante

La factorisation conditionnelle est mathématiquement exacte : toute distribution jointe peut se décomposer ainsi.

La politique plate possède déjà implicitement :

\[
p(c)=\sum_d p(c,d),\qquad
p(d\mid c)=\frac{p(c,d)}{p(c)}
\]

**Les diagnostics “case” et “chiffre” sont donc possibles dès maintenant, sans changer d’architecture.**

Ce qui change avec deux têtes apprises, c’est la paramétrisation et les gradients. Une petite tête case linéaire n’est pas garantie de représenter toutes les marginales produites par la politique plate avec le même corps CNN.

### Biais initial : point crucial pour la comparaison

Notons `k_c` le nombre de chiffres légaux dans une case.

**Politique plate initialisée à zéro :**

\[
p(c)=\frac{k_c}{\sum_j k_j}
\]

Elle visite davantage les cases ayant beaucoup de candidats.

**Politique factorisée avec deux têtes nulles :**

\[
p(c)=\frac1{N_{\text{cases disponibles}}},\qquad
p(d\mid c)=\frac1{k_c}
\]

Ce n’est pas le même hasard.

Sur les 512 grilles de référence, j’ai calculé la probabilité moyenne d’un **premier coup correct**, sans entraîner :

| Politique initiale | Probabilité |
|---|---:|
| Uniforme parmi les couples légaux | **29,25 %** |
| Case uniforme, puis chiffre légal uniforme | **33,21 %** |

**Un gain existe donc déjà avant tout apprentissage.** Il ne provient ni de labels ni d’une supervision MRV, mais d’un changement de distribution.

Il faudra distinguer :
- le gain du biais initial ;
- le gain acquis pendant l’apprentissage.

Pour une comparaison strictement contrôlée, une initialisation des logits case avec un décalage `log(k_c)` peut retrouver le hasard plat. C’est une correction de distribution issue du masque, pas une cible experte.

### Avantages et risques

| Avantages possibles | Limites et risques |
|---|---|
| Séparation explicite du choix de case et du chiffre | Amélioration non garantie |
| Paramétrisation susceptible de faciliter l’optimisation | Un mauvais chiffre pénalise aussi la case choisie |
| Faible surcoût en paramètres | Certaines cases peuvent être délaissées trop tôt |
| Diagnostics plus lisibles | Répétitions et dépendance à l’oracle inchangées |
| Compatible avec les contraintes et REINFORCE | Deux têtes peuvent aussi saturer |

Pour l’entropie, la quantité comparable à celle de la politique plate est :

\[
H(C,D)=H(C)+\sum_c p(c)H(D\mid c)
\]

Il ne faut pas additionner arbitrairement deux moyennes d’entropie.

Enfin, choisir la case la plus probable puis son chiffre le plus probable **n’est pas nécessairement** choisir le couple globalement le plus probable. Le mode glouton devra être défini précisément.

## 6. Données et protocole de comparaison

### Ce qui est déjà solide

Vérifications sur les fichiers présents :

- 1 000 grilles dans le mini-pool ;
- 200 000 dans le complet ;
- 512 au test, 128 par difficulté ;
- aucun doublon exact dans le test ;
- aucune grille exacte commune entre mini-pool et test ;
- aucune solution complète identique commune entre ces deux ensembles ;
- paires puzzle/solution valides selon le validateur existant.

Les grilles viennent d’une banque externe, pas d’une génération nouvelle à chaque épisode. Le solveur local calcule les solutions utilisées par l’oracle ; ses candidats et sa stratégie ne sont pas envoyés au réseau.

Sources : `tools/dataset.py`, `tools/sudoku_utils.py`, `tools/protocole.py:50–77`.

L’unicité des solutions est annoncée par la source, **pas vérifiée par un comptage local des solutions**. Les éventuelles équivalences par symétries n’ont pas été exclues par ces contrôles.

### Limites actuelles du protocole

- Les tests intégrés à `06_reinforce.py` et ceux de `06b`/`06c` proviennent du pool d’entraînement.
- `--complet` charge aussi les grilles réservées au test : leur exclusion n’est pas appliquée dans ce mode.
- Aucun ensemble de validation distinct n’est prévu dans ce parcours.
- Les identifiants du test sont des numéros de lignes : une reconstruction ou un réordonnancement du dataset peut changer leur sens.
- Les checkpoints sauvegardent seulement les poids, sans configuration complète, état d’optimiseur ou empreintes des données.
- `06e_reference.py` agrège tous les checkpoints fournis dans un même groupe : lui passer les deux architectures ensemble mélangerait leurs résultats.
- Les moyennes par graine de tirage et résultats par grille ne sont pas archivés par ce script, ce qui limite l’analyse statistique ultérieure.

### Référence utilisable, sans extrapolation historique

Les journaux du nouveau protocole donnent :

| Agent | Coups | Résolution |
|---|---:|---:|
| Hasard masqué | 91,6 | ≈ 100 % |
| Politique plate, 4 graines | **80,0 ± 1,2** | **99,4 % ± 0,2 point** |
| MRV avec oracle | 56,1 | 100 % |

Les dispersions du réseau sont des **écarts-types entre entraînements**, pas des intervalles de confiance.

Ces mesures forment une meilleure référence que les essais anciens. Elles ne permettent pas de déclarer directement un gain ou une régression par rapport aux 77–79 coups historiques.

### Protocole recommandé

1. **Figer les conditions.** Même mini-pool, grilles originales (`--vides 0`), horizon 150, lot 64, 500 mises à jour, Adam, `lr=3e-3`, `gamma=0`, aucun refus visible, aucun filtrage des refus, aucun bonus ajouté. Archiver configuration, version du code et empreintes des données.

2. **Contrôler l’initialisation.** Même corps CNN initial par paire de graines. Mesurer les deux hasards sans apprentissage ; annoncer si la comparaison principale conserve ou neutralise le biais « case uniforme ».

3. **Séparer validation et test.** Validation pour choisir un checkpoint ou un réglage ; test réservé à la comparaison annoncée. Les 512 grilles actuelles étant déjà consultées, elles ne doivent pas devenir le support de réglages successifs présentés comme une validation finale indépendante.

4. **Comparer par paires de graines.** Au minimum les quatre graines prévues. Séparer les flux aléatoires des données, des actions et des opérations d’optimisation : deux tirages successifs consomment autrement l’aléatoire qu’un seul.

5. **Enregistrer les résultats individuels.** Par architecture, graine d’entraînement, grille et tirage. Comparer les différences appariées avec leur incertitude ; ne pas traiter les répétitions d’une même grille comme des grilles indépendantes.

6. **Mesurer efficacité et fiabilité ensemble.** Coups avec échecs comptés à 150, taux de résolution, faux, répétitions, cases restantes au timeout, résultats par difficulté, temps et nombre de transitions. Même nombre d’itérations ne signifie pas même coût de calcul.

7. **Diagnostiquer les deux politiques de la même façon.** Distribution des cases choisies selon `k`, réussite conditionnelle au chiffre, entropie jointe et composantes, saturation des logits. Aucun diagnostic ne devient une cible d’apprentissage.

Les douze tests prévus dans le plan sont pertinents. J’ajouterais les contrôles d’égalité de distribution initiale, de reconstruction des probabilités jointes et de comportement aux masques vides.

**Le seuil de 65 coups est un objectif pédagogique, pas une condition nécessaire pour démontrer l’intérêt de la factorisation.** Un gain robuste plus petit resterait un résultat utile. Inversement, une moyenne sous 65 ne certifierait pas un solveur autonome sans oracle.

## 7. Références documentaires : vérification critique

| Référence | Vérification et pertinence |
|---|---|
| [Sudoku Exchange Puzzle Bank](https://github.com/grantm/sudoku-exchange-puzzle-bank) | Source pertinente : génération QQWing, unicité annoncée, classement Sukaku Explainer, domaine public. Les niveaux décrivent une difficulté de résolution logique, pas nécessairement celle de notre environnement à essais validés. |
| [Avra98/sudoku-gpt2-curriculum](https://huggingface.co/Avra98/sudoku-gpt2-curriculum) | La fiche décrit explicitement des cibles supervisées de candidats, SFT puis GRPO, et un modèle préentraîné. **Incompatible comme recette avec tes contraintes.** Inspiration limitée au curriculum. |
| [Avra98/sudoku-latent-backtracking](https://huggingface.co/Avra98/sudoku-latent-backtracking) | Même problème de supervision. Le « backtracking » décrit concerne notamment la révision d’étapes antérieures du curriculum, pas simplement l’annulation d’un mauvais placement Sudoku. L’analyse locale entretient une ambiguïté. |
| [AdaC-GRPO, arXiv:2610.13184](https://arxiv.org/html/2610.13184v1) | Article vérifié. Les chiffres 0,405/0,586 et 0,237/0,586 figurent bien dans les ablations. Mais il s’agit de modèles multimodaux, avec initialisation supervisée sur 300 traces et récompense de solution complète. **Ni preuve pour la factorisation, ni justification directe de nos récompenses par coup.** |

Le document `docs/analyse_complete_sudoku_rl_2026-11-13_1401.md` recommande encore supervision de candidats, solutions et mémoire des refus. **Il est en conflit avec la trajectoire actuelle** et doit être lu comme une analyse historique dépassée, pas comme une spécification.

Le plan `docs/plan_agent_sudoku_generique_sans_labels.md` est globalement cohérent avec tes contraintes. En revanche, l’affirmation selon laquelle la représentation de l’action serait le blocage principal reste une **hypothèse**, pas un diagnostic établi.

## 8. Recommandation

**Avis favorable à une expérience factorisée contrôlée ; avis défavorable à la présenter comme la solution démontrée au plateau.**

Avant l’entraînement comparatif, les priorités sont de préciser le contrat « joueur avec oracle », verrouiller les conditions sans refus ni fuite de test, traiter les cas limites des masques et distinguer le biais initial de l’apprentissage.

Conserver ensuite le CNN, REINFORCE et les récompenses pour cette seule comparaison. Ni curriculum, ni critique, ni supervision supplémentaire à ce stade.

**La question expérimentale juste est : « À observations, règles, récompenses et budget contrôlés, cette paramétrisation apprend-elle une meilleure politique que la politique plate ? »** Un échec ne prouverait ni que REINFORCE est définitivement limité, ni qu’un actor-critic est automatiquement la prochaine solution.
