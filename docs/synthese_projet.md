# Synthèse du projet mini-llm

Projet de découverte : construire un petit GPT à la main, du neurone au chat, et mesurer comment le corpus d'entraînement change le modèle. Le détail des essais est dans `suivi_experiences_corpus.md`.

## Ce qui a été fait

| Chapitre | Contenu |
|---|---|
| `01_fundamentals` | neurone, rétropropagation, XOR, en NumPy puis PyTorch |
| `02_tokenizer` | caractères, mots, BPE |
| `03_transformer` | attention, bloc, petit GPT en NumPy |
| `04_training` | GPT PyTorch, boucle d'entraînement, checkpoints |
| `05_inference` | génération, température, Top-K, Top-P |
| `06_corpus` | 346 livres : nettoyage, BPE à 4 612 tokens, entraînements, mesure complète, théâtre, chat |

## Résultats principaux (perte sur la validation des livres, plus bas = mieux)

| Modèle | Changement | Perte rapide | Perte complète |
|---|---|---|---|
| référence | T = 128 | 3,321 | |
| B | contexte 256 | 3,227 | 3,1815 |
| A | d = 384, 6 blocs | 3,068 | 3,0156 |
| D | corpus filtré | 3,106 | 3,0497 |
| P4 | 8 blocs | 3,033 | 2,9792 |
| P4 + théâtre | + 6,7 % de dialogues | 3,024 | 2,9701 |

Sur les livres, P4 + théâtre (2,9701) ne perd presque rien face à P4 (2,9792) : le théâtre ajouté ne dégrade pas les romans (écart −0,009, sous le bruit). Sur les dialogues jamais appris, P4 + théâtre gagne 0,145 en mesure complète (2,8361 → 2,6913 ; 1,415 → 1,343 bit/octet), soit six fois l'écart P4 − A.

## Ce qu'on a appris
- **Le contexte et la largeur paient** : T 128 → 256 et d 256 → 384 donnent les plus gros gains. Passer de 6 à 8 blocs coûte 25 % de temps pour environ −0,03.
- **Le corpus compte** : un filtrage a aidé moins que prévu (D plus mauvais que A) ; le théâtre aide surtout là où on le mesure (les dialogues).
- **La mesure compte autant que le modèle** : la validation rapide et la complète ne sont pas comparables (environ −0,05 d'écart). Toujours comparer des mesures du même type.
- **Le bruit** : environ 0,03. En dessous, un écart n'est pas démontré.
- **Le sens ne vient pas** à cette échelle (17,8 M de paramètres) : le modèle imite le style, pas le contenu. Voir la section suivante pour savoir à partir de quand il apparaît.

## À partir de quand le sens apparaît-il ? (ordres de grandeur)

Ce qui suit vient de publications connues, citées de mémoire : les chiffres sont des ordres de grandeur à vérifier avant de les republier.

**Ce que les chercheurs ont constaté**
- **La perte baisse régulièrement** (Kaplan et coll., 2020) : plus de paramètres, de données et de calcul, et la perte diminue selon une loi de puissance, sans palier. Cette courbe-là était prévisible.
- **Les capacités, elles, semblent apparaître par marches** (« capacités émergentes », Wei et coll., 2022) : en dessous d'une certaine taille, une tâche (arithmétique, questions à quelques exemples) reste au hasard, puis elle devient réussie. Les tâches de ce type apparaissaient pour des modèles de l'ordre de la dizaine à la centaine de milliards de paramètres.
- **La surprise** : GPT-2 (2019, 1,5 milliard de paramètres) puis GPT-3 (2020, 175 milliards) ont fait des choses pour lesquelles ils n'avaient pas été entraînés (traduire, répondre, suivre des exemples). Les chercheurs n'avaient pas prédit ces capacités, et le tâtonnement a été réel : on a essayé plus grand parce que la perte baissait, pas parce qu'on savait que le sens viendrait.
- **Le doute** (Schaeffer et coll., 2023) : une partie des « marches » vient de la façon de mesurer (réussi ou raté, tout ou rien). Avec une mesure plus fine, la progression paraît souvent continue. L'effet escalier est donc en partie réel, en partie une illusion de métrique. Le débat reste ouvert.
- **Le sens avec de petits modèles** (TinyStories, Eldan et Li, 2023) : des modèles de 1 à 30 millions de paramètres produisent des histoires cohérentes, à condition que le corpus soit très simple (vocabulaire d'enfant de 4 ans). Notre modèle a une taille comparable, mais lit des romans d'adultes, beaucoup plus riches.
- **Le bon équilibre** (Chinchilla, 2022) : environ 20 tokens d'entraînement par paramètre. Nous sommes à environ 4 (71 millions de tokens pour 17,8 millions de paramètres), donc notre modèle est même en manque de données pour sa taille.

**Où nous nous situons**

| | Paramètres | Calcul d'entraînement (approx. 6 × paramètres × tokens vus) |
|---|---|---|
| mini-llm (P4) | 17,8 M | environ 2 × 10^16 opérations |
| GPT-2 | 1,5 Md | non retenu ici |
| GPT-3 | 175 Md | environ 3 × 10^23 opérations |

Nous sommes donc environ **dix millions de fois en dessous du calcul de GPT-3**, et environ 100 fois moins de paramètres que GPT-2 (en plus, avec bien moins de données).

**Estimation prudente pour notre cas** (une estimation, pas une mesure)
- Texte **grammatical et cohérent sur une phrase** : atteint (la perte 3,0 donne déjà ça).
- Texte **cohérent sur un paragraphe** avec un vocabulaire riche : probablement à partir de 100 à 500 millions de paramètres, avec des milliards de tokens.
- **Répondre à des questions** à partir d'exemples : plutôt au-delà du milliard de paramètres, ou bien un modèle plus petit affiné sur des paires question-réponse.
- **Raisonner, calculer** : plutôt à partir de dizaines de milliards de paramètres.
- Pour une GTX 1070 : un corpus simple (type TinyStories) donnerait plus de sens qu'un modèle plus gros sur des romans. C'est une expérience à notre portée.

**Ce que cela nous apprend sur notre projet**
- L'absence de sens n'est pas une erreur de code : c'est l'échelle et la difficulté du corpus.
- La perte est la bonne mesure pour progresser (elle varie sans palier). Les capacités, elles, se testent avec un jeu de questions fixe.
- Sources à consulter : Kaplan 2020 (« Scaling Laws for Neural Language Models »), Hoffmann 2022 (« Training Compute-Optimal Large Language Models », Chinchilla), Wei 2022 (« Emergent Abilities of Large Language Models »), Schaeffer 2023 (« Are Emergent Abilities of Large Language Models a Mirage? »), Eldan et Li 2023 (« TinyStories »).

## Limites
- Un seul entraînement par essai, une seule graine.
- Validation de dialogues petite (quelques pièces).
- Le chat n'a pas été évalué par un jeu de questions fixe.
- Corpus Claire sous licence CC BY-NC-SA : ne pas republier. Les livres sont sous droit d'auteur : jamais dans le dépôt.
- Image `img/FB_IMG_…jpg` : droits à vérifier. Pas de `LICENSE` pour l'instant.

## Ce qu'il reste à explorer
1. Affinage (fine-tuning) sur des paires question-réponse, avec jeu de questions fixe.
2. P3 : 8 têtes d'attention (non lancé).
3. Conversations libres de Claire (ESLO, OFROM, PFC, LINAGORA) avant l'affinage.
4. Répéter un essai avec 2 ou 3 graines pour chiffrer le bruit.
5. Précision mixte (fp16) pour accélérer sur la GTX 1070.
6. Nouveau projet : radar Arduino (profil de 150 distances par balayage), avec un MLP ou une convolution 1D, après enregistrement de données.

## Pour reprendre
- Avant tout entraînement : `REPRENDRE = False` et un nouveau `NOM` dans `07_entrainement_corpus.py`.
- Nouveau dossier de données : l'ajouter à `.gitignore` avant de le créer.
- Une seule variable à la fois, et mêmes mesures pour comparer.
