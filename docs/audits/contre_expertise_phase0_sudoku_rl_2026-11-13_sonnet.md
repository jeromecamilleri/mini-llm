# Contre-expertise de l'audit d'Astra

> **Auteur : Claude Sonnet 5.5 (identifiant de modèle : `claude-sonnet-5.5`)**, via GitHub Copilot CLI, le 13 novembre 2026.
> Contre-expertise de l'audit de GPT-6 Astra (`gpt-6-astra`) : `docs/audit_phase0_sudoku_rl_2026-11-13_astra.md`.
> Aucun fichier modifié, aucun entraînement lancé. Tout ce qui suit est mesuré sur CPU par des scripts lancés depuis le terminal, sans rien écrire sur le disque.

Pour dépasser le rapport, j'ai vérifié trois choses : l'unicité des solutions, la tête de valeur, et la partie jouée sans oracle. J'ai aussi remplacé le calcul d'Astra sur le premier coup par une simulation de partie complète.

**Verdict.** Le rapport d'Astra est exact sur les faits de code. Mais il sous-estime de beaucoup un biais qui peut invalider la comparaison, et il ne regarde pas ce que le réseau a vraiment appris.

## 1. Oracle et environnement

**Confirmé.** `env_lots.py:step` (ligne 85) n'accepte un coup que si `oui = actif & vide & ~doublon & bon`. Seul `oui` modifie `grille`, `ligne_a`, `col_a` et `bloc_a`. Un coup légal mais faux laisse la grille inchangée.

J'ai testé sur 64 grilles :
- 69,2 % des couples légaux de l'état initial sont faux ;
- tous ces coups donnent −1 et ne changent rien ;
- le chiffre de la solution d'une case vide est toujours légal : le nombre de coups corrects égale le nombre de cases vides.

**Ce que ça change.**
- La partie est une suite de questions à l'oracle, avec des essais gratuits à −1.
- Aucune erreur ne se propage dans la grille, donc il n'y a jamais de retour arrière à apprendre.
- La règle n'est pas apprise : le masque la fournit (`reseau.py:masques_actions`, ligne 27).

**Incohérence documentaire.** `README.md:5` dit que « le réseau découvre progressivement les règles du Sudoku en recevant des récompenses ». C'est faux : `fiches/03_mesures.md` dit lui-même que le masque est « un choix de conception, pas de l'apprentissage ».

**Mesure absente du rapport d'Astra : la partie sans oracle.** Les coups légaux sont acceptés tels quels, sur les 512 grilles de test :

| Agent | Grilles exactes | Cases fausses par grille |
|---|---:|---:|
| Hasard plat | 0 % | 23,3 |
| Hasard factorisé | 0 % | 23,4 |
| **Réseaux de référence g0 à g3 (argmax ou tirage)** | **0 %** | **23,8 à 24,5** |
| MRV, qui devine quand k ≥ 2 | 29,0 % | 15,9 |
| Chiffres uniques seuls (k = 1) | 18,4 % | — |

Le réseau n'a donc acquis aucune connaissance des chiffres : sans oracle, il se trompe autant que le hasard. Ce qu'il a appris, c'est de mieux utiliser l'oracle, et en partie de mieux choisir les cases.

**Cohérence avec l'objectif.** Le contrat documenté est respecté (`oracle.py` en tête de fichier, `sudoku_env.py:82`). Mais « jouer au Sudoku » sans oracle, comme dans l'application mobile de la Phase 4, n'est pas atteint. Je ne changerais pas l'environnement maintenant. Il faut en revanche ajouter cette évaluation sans oracle : elle ne touche pas à l'entraînement.

## 2. REINFORCE et GAMMA = 0

**Astra est exact** sur `retours` (`reinforce.py:57`) et sur `pas_reinforce` (lignes 78 à 80, `optimiseur.step()` ligne 103).

Il y a un point qu'il ne souligne pas. Avec l'oracle, le nombre de coups corrects restants est fixe. Le retour non actualisé vaut donc une constante moins le nombre de coups faux futurs. `GAMMA = 0` n'est alors qu'un substitut myope de l'objectif réel, et seul l'ordre des cases les sépare. Le coup qui maximise la précision immédiate (1/k) est celui de MRV. Je juge donc ce choix cohérent avec l'objectif, et pas seulement acceptable. Les essais 8 et 8b le confirmaient déjà : un gain sur 4 graines sur 4.

**Biais de baseline.** Je les trouve négligeables :
- moins de 5 parties actives : 86 coups sur 4 983 (1,7 %) ;
- le facteur (1 − 1/n) vaut 0,98 pour 64 parties actives.

**Point qu'Astra n'a pas relevé.** Le bonus de victoire de +10 gonfle l'écart-type de normalisation :
- avec le bonus : 1,43 ;
- sans le bonus : 0,90.

Les avantages sont donc réduits d'environ 37 %. Adam compense en partie, et je ne l'ai pas testé plus loin.

**Autre incohérence.** La docstring de `06_reinforce.py` (lignes 17 et 18) décrit encore GAMMA = 0,99 et une moyenne globale des retours. Le code utilise une base par numéro de tour. De plus, les valeurs par défaut du script (`--gamma 0.99` ligne 62, `--lr 3e-4` ligne 60) diffèrent de la configuration de référence (`--gamma 0`, `--lr 3e-3`). Quelqu'un qui lance le script sans options ne reproduira pas la référence.

## 3. Tête de valeur

**Inutilisée : confirmé par une expérience.** Après un `pas_reinforce` suivi d'un pas d'Adam sur un réseau neuf :
- les deux `grad` de `tete_valeur` valent `None` ;
- ses poids sont strictement identiques ;
- la tête politique a bien changé.

**Raisons de la conserver.**
1. Le chargement strict échoue sans elle. Tous les scripts passent par `charger_modele` (`reseau.py:87`, ligne 91), ce qui casserait les 4 checkpoints de référence.
2. Elle n'a aucun effet sur la dynamique d'entraînement.
3. Elle servira de base apprise en Phase 5, une baseline dépendant de l'état, ce qui réduit la variance sans biais.
4. Garder les mêmes paramètres inutilisés dans les deux bras garde la comparaison équitable.

**Recommandation.** La garder, et la documenter comme « non entraînée ». Corriger `fiches/04_reseau.md:68`, qui laisse croire qu'elle réduit le bruit aujourd'hui. Compter 301 833 paramètres utiles et non 301 898.

## 4. Politique factorisée

### Ce qui est correct

- **Décomposition et log-probabilité.** La règle de la chaîne, la somme des deux log-probabilités et la compatibilité avec REINFORCE sont exactes.
- **Entropie jointe.** H(C) + Σ p(c) H(D|c) est la bonne quantité.
- **Argmax.** Le meilleur couple n'est pas la meilleure case suivie de son meilleur chiffre.
- **Taille de la nouvelle tête.** Elle ajoute 65 paramètres, la tête actuelle en comptant 585.

### Ce que la factorisation n'apporte pas

Elle n'ajoute aucune expressivité significative. Elle change la normalisation du softmax et donne un logit propre à la case. Le gain éventuel viendrait de l'optimisation, et non d'une capacité nouvelle.

### 33,21 % contre 29,25 % : calcul correct, interprétation insuffisante

Je retrouve exactement 0,2925 et 0,3321. De plus, l'inégalité arithmético-harmonique donne « factorisé ≥ plat » sur chacune des 512 grilles (vérifié). Mais ce chiffre ne concerne que le premier coup.

J'ai donc simulé les deux hasards sur des parties complètes, avec oracle :

| Agent | Coups | Faux | Répétitions |
|---|---:|---:|---:|
| Hasard plat | 91,4 | 38,8 | 9,1 |
| **Hasard factorisé** | **82,8** | 30,2 | 5,6 |
| Réseau plat entraîné (référence) | 80,0 ± 1,2 | 27,6 | 14,7 |
| MRV | 56,1 | 3,5 | 0,5 |

Les 91,4 et 82,8 viennent d'un générateur CPU, d'où un léger écart avec les 91,6 de la référence mesurée sur GPU.

**Un hasard factorisé, sans aucun apprentissage, est déjà à 2,8 coups du réseau plat entraîné.** Le biais initial vaut environ 8,6 coups, soit sept fois la dispersion entre graines. Astra signalait le risque sans le chiffrer, et c'est ce qui change tout pour la comparaison.

**Interprétation.** Un softmax plat donne à chaque case une masse proportionnelle à k, le nombre de chiffres légaux. Il favorise donc les cases les moins contraintes.

**Mise en garde sur la décomposition du gain.** Je n'ai pas établi comment le gain du réseau plat (91,4 → 80,0) se répartit entre le choix de la case et celui du chiffre. Ce que j'ai mesuré : la probabilité de jouer un chiffre unique, quand il en existe un, ne monte que de 0,34 à 0,38 (MRV : 1,00, hasard factorisé : 0,42). Cela ne prouve pas que le réseau n'a rien appris sur les cases.

**Imprécision d'Astra.** Il propose d'« initialiser » les logits de case avec un décalage log(k_c). Une initialisation de poids ne peut pas le faire, car k_c dépend de l'état. C'est un terme architectural permanent calculé depuis le masque, donc autorisé mais à valider.

### Comparaison équitable

Un plan 2 × 2 permet d'isoler la paramétrisation :

| | Départ : cases au prorata de k | Départ : cases uniformes |
|---|---|---|
| **Plate** | A : référence | D : plate avec décalage −log k_c, **sans nouvelle tête** |
| **Factorisée** | C : factorisée avec décalage +log k_c | B : factorisée naturelle |

D est une ligne de code et teste directement l'hypothèse du biais en k. Si D atteint le niveau de B, la tête de case n'est pas nécessaire. Le décalage −log k_c rend la sélection des cases uniforme, il ne favorise pas MRV.

## 5. Protocole expérimental

**Astra est suffisant sur** la séparation validation/test, le jeu de test fixe, la publication des échecs et les mesures par difficulté. L'unicité des solutions était signalée comme « non vérifiée localement » : je l'ai vérifiée. Les 512 grilles de test et les 1 000 du pool ont exactement une solution. Mon compteur trouve ≥ 2 solutions sur les 20 grilles appauvries de contrôle, donc il détecte bien la non-unicité.

**Biais qui pourraient invalider la comparaison.**
1. **Le biais initial** de 8,6 coups, voir plus haut : le 2 × 2 est obligatoire, pas optionnel.
2. **Les hyperparamètres** (`lr` 3e-3, 500 itérations) ont été ajustés sur le bras plat. Le bras factorisé hérite d'un réglage qui n'est pas le sien. Il faut un balayage identique pour les deux, sur validation.
3. **Seules les poids finaux sont sauvegardés** (`06_reinforce.py:151`). Avec `lr` 3e-3, la politique fluctue d'une itération à l'autre : sauvegarder aussi des points intermédiaires permettrait de comparer des courbes sur le protocole de test.
4. **Puissance statistique.** La dispersion entre graines (1,2) domine :
   - le tirage seul donne une erreur-type de 0,75 sur 512 grilles pour un tirage, et 0,37 entre 4 tirages ;
   - environ 90 % de la variance vient de l'entraînement.
   - Mon estimation (non mesurée), avec un test t à 80 % de puissance et un seuil de 5 % : la différence minimale détectable est d'environ 2,7 coups avec 4 graines par bras, et 1,7 avec 8. Il faut donc investir dans les graines, pas dans les grilles ni les tirages.
5. **Métriques redondantes.** Pour une grille résolue, coups = cases vides + coups faux : 80,0 − 27,6 = 52,4 ; 56,1 − 3,5 = 52,6 ; 91,6 − 39,0 = 52,6. Il suffit de rapporter les coups faux par grille.
6. **Les répétitions** mesurent la concentration de la politique, pas un manque de mémoire. Elles baissent mécaniquement avec le hasard factorisé, de 9,1 à 5,6.

**Mesures pertinentes à ajouter**, toutes en évaluation seule :
- la part des coups faux par valeur de k, comparée à 1 − 1/k ;
- la probabilité de jouer un chiffre unique quand il y en a un, en mesure et jamais en cible ;
- la résolution sans oracle ;
- l'entropie jointe.

**Garde-fous.** Une règle d'arrêt déclarée à l'avance en cas de divergence, comptée comme échec. Les seuils de décision (effet minimal et nombre de graines) fixés avant les essais.

**Point où Astra est trop prudent.** Il recommande de séparer les flux aléatoires des données, des actions et de l'optimisation. Les données sont déjà isolées (le `rng` NumPy dans `tirer`). Et l'ordre des paquets ne compte pas, car `pas_reinforce` accumule les gradients avant un seul pas d'Adam.

## Synthèse

**D'accord avec Astra sur :**
- l'architecture et son champ de vue ;
- les formules et le calcul des masques ;
- la tête de valeur inutilisée ;
- les problèmes de pool et d'ensemble de test des anciens essais ;
- la nuance sur la saturation des logits ;
- la prudence sur les références externes, notamment l'incompatibilité des dépôts Avra98 avec nos contraintes ;
- la recommandation générale de comparer sans changer REINFORCE ni les récompenses.

**Erreurs et lacunes d'Astra :**
- il ne chiffre pas le biais initial : 8,6 coups pour le hasard factorisé, soit près de l'écart entre le hasard plat et le réseau entraîné ;
- il parle d'« initialisation » log(k_c), alors qu'il faut un terme architectural dépendant de l'état ;
- il laisse l'unicité des solutions non vérifiée ;
- il ne mesure pas la résolution sans oracle ;
- il ne relève pas la docstring périmée ni les valeurs par défaut de `06_reinforce.py` ;
- il ne relève pas l'effet du bonus de victoire sur la normalisation.

**À trancher par expérience :**
1. L'effet de la factorisation une fois le biais initial neutralisé (le plan 2 × 2).
2. La répartition du gain du réseau plat entre le choix de la case et celui du chiffre.
3. L'effet du `lr` sur chaque bras.
4. L'intérêt d'une base apprise par la tête de valeur (Phase 5).
5. L'effet de `GAMMA` entre 0 et 1, plus tard.
6. Le bonus de victoire, de faible priorité.
7. Si un entraînement peut obtenir des grilles exactes sans oracle.

## Recommandation

1. **Avant d'écrire la Phase 2**, étendre l'évaluation : coups faux par grille, résolution sans oracle, diagnostic par k, hasard factorisé comme baseline, et sauvegardes intermédiaires. Rien de tout cela n'est écrit à ce jour.
2. **Lancer d'abord D**, la politique plate avec décalage −log k_c. C'est une seule ligne, elle teste le biais en k sans nouvelle architecture, et elle départage les hypothèses avant d'investir dans les deux têtes.
3. **Puis le 2 × 2 complet**, avec 4 graines par bras au début, puis 8 si l'effet reste entre 1,5 et 3 coups. Garder la tête de valeur, REINFORCE et les récompenses inchangés.
4. **Déclarer d'avance** que la factorisation n'est utile que si B dépasse D d'au moins 3 coups.
5. **Précision à valider** : le décalage mesuré depuis le masque respecte toutes les contraintes (aucun label, aucune solution, aucune mémoire des refus). Mais comme il s'agit d'un choix de conception, il doit être soumis avant d'être codé.
