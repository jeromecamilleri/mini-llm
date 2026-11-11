# Suivi des expériences : GPT sur les 346 livres

Règle : **une seule variable à la fois**. Mêmes réglages de génération pour comparer (début « Il était une fois », graine 0, température 0,8). Pas à pas du corpus : [`howto_corpus_353_livres.md`](howto_corpus_353_livres.md).

## Fiche technique de référence (expérience 0)

| Élément | Valeur |
|---|---|
| Vocabulaire | 4 612 tokens (BPE, 4 000 fusions) |
| Dimension d | 256 |
| Blocs | 4 (4 têtes d'attention, 64 par tête) |
| Réseau interne (FFN) | 256 → 1024 → 256, GELU |
| Contexte T | 128 tokens (environ 90 mots) |
| Paramètres | 5 553 664 (dont 2 361 344 d'embeddings, 43 %) |
| Entraînement | batch 32, 20 000 étapes, 1,24 passage sur 66 M de tokens |
| Taux d'apprentissage | 1e-3 → 1e-4 (échauffement 300 + cosinus), AdamW, poids 0,1, clip 1,0 |
| Données | 329 livres d'entraînement (66 216 164 tokens), 17 de validation (3 046 367) |
| Matériel | GTX 1070, 1,4 Go de VRAM utilisés sur 8,6 Go |

## Tableau de suivi

| # | Variable changée | Paramètres | Durée | Validation (perplexité) | Train | Texte / remarque | État |
|---|---|---|---|---|---|---|---|
| 0 | référence | 5,55 M | 14 min | 3,321 (28) | 3,141 | phrases plausibles, sans fil ; boucle à T 0,3 ; déraille à T 1,2 | fait |
| B | contexte T = 256 (reste inchangé ; batch 32 : 8 192 tokens par étape, 2,47 passages, donc deux fois plus de données vues) | 5,59 M | 30 min | 3,227 (25) | 3,045 | phrases plausibles, noms propres plus nombreux ; encore pas de fil net (à vérifier avec la leçon 8) | fait, textes à comparer |
| A | modèle plus gros, cumulatif sur B : d = 384, 6 blocs, 6 têtes (version intermédiaire de d = 512 / 8 blocs, ~4 h) ; LR 6e-4 | 14,29 M | 74 min | 3,068 (21) | 2,842 | phrases plus soutenues, dialogues et noms propres cohérents sur quelques lignes ; encore pas de fil | fait |
| D | texte d'entraînement filtré (leçons 11-12) : 1,93 % des caractères retirés, 64,3 M tokens ; validation inchangée | 14,29 M | 74 min | 3,106 (22) sur la validation non filtrée ; **3,078 sur la validation filtrée, à égalité avec A** (3,079) | 2,871 | pas de gain mesurable ; l'écart de 0,038 avec A vient du bruit de la validation (leçon 14) | fait |
| C | plus d'étapes (60 000) | 5,55 M | environ 45 min | | | | à faire |
| E | réglages (taux, dropout) | 5,55 M | | | | | à faire |

Les durées et paramètres des expériences à faire sont des estimations, pas des mesures.

## Analyse de la situation (11 novembre 2026)

Constat : le texte est correct phrase par phrase (orthographe, conjugaison, dialogues) mais sans cohérence d'une phrase à l'autre. Le chat répond « n'importe quoi ».

- **Corpus** : pas le principal coupable. 66 M de tokens, train 3,14 et validation 3,32 : écart de 0,18, pas de surapprentissage.
- **Taille** : 5,5 M de paramètres, dont près de la moitié d'embeddings ; il reste environ 3 M pour « comprendre ». GPT-2 en a 124 M.
- **Contexte 128** : le modèle ne voit que environ 90 mots, trop peu pour suivre un personnage ou une histoire. Facteur important.
- **Durée** : 1,2 passage, courbe presque plate à la fin (gain de 0,001 par tranche de 500 étapes) : continuer tel quel rapporte peu.
- **Bruit** : tables des matières (« 2273-Oastronomie… »), mentions d'éditeur, 611 caractères dont beaucoup de rares.
- **Chat** : le modèle n'a jamais vu de paires question → réponse ; il faudrait un second entraînement (ajustement par instructions).

## Mémoire GPU

On utilisait 1,4 Go sur 8,6. Augmenter T coûte cher : l'attention grandit comme T² (T = 512 : 16 fois plus d'attention qu'à 128). Un modèle plus gros utilise aussi plus de mémoire. Ordres de grandeur non mesurés.

## Méthode : expériences cumulatives

Chaque expérience ne change qu'**une variable par rapport à l'expérience précédente retenue**.

| Modèle | Contexte | Taille | Comparé à |
|---|---|---|---|
| référence | 128 | 5,5 M | |
| B | 256 | 5,5 M | référence |
| A (cumulatif) | 256 | environ 30 M | B |
| A (séparé) | 128 | environ 30 M | référence |

Décision du 11 novembre : on enchaîne en cumulatif. On ne garde T = 256 que si B améliore les textes ou la perte ; sinon A repart de T = 128. Choix de batch pour B : 32 (décision de Jérôme) : même batch que la référence, mais 8 192 tokens par étape ; l'effet du contexte se mélange à celui de la quantité de données vues. Mémoire : la carte a 8 192 MiB (8,6 Go) ; ce qui l'utilise vraiment pour progresser, c'est la taille du modèle (expérience A), pas le batch.

## Ordre recommandé

1. **B** : T = 256 avec le même modèle, pour isoler l'effet du contexte. Nouveau nom de checkpoint pour ne pas écraser `livres_meilleur.pt`.
2. **A** : modèle plus gros, avec le meilleur contexte.
3. **D** : nettoyage, qui profite à toutes les expériences suivantes.

Attention : changer T change un peu la validation ; comparer aussi les textes générés.

## Journal des résultats

À remplir après chaque lancement : copier la sortie dans `logs/` et noter ici validation, durée et impression sur le texte.

## Architecture et réglages du réseau

| Réglage | Ce que c'est | Réf. | A |
|---|---|---|---|
| `D` | taille du vecteur de chaque token (largeur) | 256 | 384 |
| `N_BLOCS` | blocs Transformer empilés (profondeur) | 4 | 6 |
| `N_TETES` | têtes d'attention par bloc (`D` divisible par `N_TETES`) | 4 | 6 |
| neurones cachés du MLP | 4 × `D`, par bloc | 1 024 | 1 536 |
| `T` | fenêtre de contexte | 128 | 256 |
| vocabulaire | tokens du BPE | 4 612 | 4 612 |
| Paramètres | embeddings + blocs | 5,55 M | 14,29 M |

Paramètres d'un bloc : 12 D² + 13 D (attention 4 D², MLP 8 D², plus biais et LayerNorm). Total : `N_BLOCS` × bloc + 2 × vocabulaire × D (embeddings et sortie) + T × D (positions) + 2 D (LayerNorm final). Pour A : 6 × 1 774 464 + 3 542 016 + 98 304 + 768 = 14 287 872.

**Constat.** A a changé `D` (256 → 384), `N_BLOCS` (4 → 6), `N_TETES` (4 → 6) et le taux d'apprentissage en même temps : gain de 0,159, mais on ne sait pas quelle part vient de la largeur, de la profondeur ou des têtes. C'est contraire à notre règle « une seule variable à la fois », et je n'avais pas proposé de les séparer.

Essais possibles pour les séparer, à partir de B (une variable chacun, environ 30 min) :

| Essai | Changement | Question |
|---|---|---|
| P1 | `N_BLOCS` 4 → 6, le reste comme B | la profondeur aide-t-elle ? |
| P2 | `D` 256 → 384, le reste comme B | la largeur aide-t-elle ? |
| P3 | `N_TETES` 4 → 8, le reste comme B | combien de têtes ? |

## Registre : nom du modèle → réglages

Chaque entraînement écrit aussi `models/checkpoints/{NOM}_reglages.json` (tous les réglages). Le checkpoint `.pt` contient seulement la config du réseau (d, blocs, têtes, T, vocabulaire), pas le batch ni le taux. Pour la référence, qui date d'avant ce fichier, voici ses réglages :

| Nom | d | Blocs | Têtes | T | Batch | Tokens/étape | Étapes | Taux | Paramètres | Validation |
|---|---|---|---|---|---|---|---|---|---|---|
| `livres` (référence) | 256 | 4 | 4 | 128 | 32 | 4 096 | 20 000 | 1e-3 → 1e-4 | 5 553 664 | 3,321 |
| `livres_T256` (B) | 256 | 4 | 4 | 256 | 32 | 8 192 | 20 000 | 1e-3 → 1e-4 | 5 586 432 | 3,227 |
| `livres_A_d384` (A) | 384 | 6 | 6 | 256 | 32 | 8 192 | 20 000 | 6e-4 → 6e-5 | 14 287 872 | 3,068 |
| `livres_D_filtre` (D) | 384 | 6 | 6 | 256 | 32 | 8 192 | 20 000 | 6e-4 → 6e-5 | 14 287 872 | 3,106 (3,078 filtrée) |

**TAILLE_BATCH** : nombre de fenêtres lues avant chaque correction des poids ; tokens lus par étape = batch × T. Détail dans `src/06_corpus/fiches/07_entrainement_corpus.md`.

## Historique des modèles (checkpoints)

Chaque expérience écrit ses propres fichiers (réglage `NOM` dans `07_entrainement_corpus.py`) : rien n'est écrasé.

| Expérience | Checkpoint | CSV de la courbe |
|---|---|---|
| 0 référence | `models/checkpoints/livres_meilleur.pt` | `logs/entrainement_livres.csv` |
| B T = 256 | `models/checkpoints/livres_T256_meilleur.pt` | `logs/entrainement_livres_T256.csv` |

| A d = 384 | `models/checkpoints/livres_A_d384_meilleur.pt` | `logs/entrainement_livres_A_d384.csv` |
| D texte filtré | `models/checkpoints/livres_D_filtre_meilleur.pt` | `logs/entrainement_livres_D_filtre.csv` |

Pour utiliser un modèle : changer `CHECKPOINT` dans `08_generation_corpus.py` ou `10_chat.py` ; pour sa courbe, `CSV` dans `09_courbe_perte.py`.

### Expérience B : T = 256, batch 32 (11 novembre 2026)

Sortie complète : `logs/entrainement_livres_T256_sortie.txt` ; courbe : `logs/entrainement_livres_T256.csv`.

| Mesure | Référence (T = 128) | B (T = 256) |
|---|---|---|
| Paramètres | 5 553 664 | 5 586 432 (+32 768 : table des positions) |
| Tokens par étape | 4 096 | 8 192 |
| Durée | 14 min | 30 min (estimation 28-35 min : juste) |
| Validation | 3,321 (perplexité 28) | **3,227 (perplexité 25)** |
| Train | 3,141 | 3,045 |
| Écart val − train | 0,180 | 0,182 |
| Validation à l'étape 500 | 4,794 | 4,745 |

- GPU pendant B : 99 %, 153 W / 166 W, 66 °C, 1 721 MiB utilisés sur 8 192 (1 500 MiB pour l'entraînement) : la mémoire est loin d'être le facteur limitant, le calcul l'est.
- Gain de 0,094 de perte (perplexité 28 → 25). Attention : B mélange deux effets (contexte doublé et deux fois plus de données vues), et la validation porte sur des fenêtres deux fois plus longues. On ne peut donc pas dire quelle part vient du contexte.
- Même écart val − train que la référence : pas de surapprentissage, même avec 2,5 passages.
- La validation baissait encore à la fin (gain de 0,003 par tranche de 500 étapes), moins plat que la référence : plus d'étapes aideraient un peu.
- Les textes : à comparer avec la leçon 8 (`CHECKPOINT = "livres_T256_meilleur.pt"`).
- Le chat (`10_chat.py`) pointe maintenant sur ce modèle.

### Expérience D : préparation (11 novembre 2026)

- Audit du corpus : 19 122 lignes `@font-face { }` et des milliers d'autres lignes de feuilles de style, ISBN, mentions d'éditeur, codes de mise en page (`FL4103`), alphabets rares (611 caractères différents).
- Filtrage (leçon 11) : 1,93 % des caractères retirés, écrits dans `datasets/cleaned_v2/` (`cleaned/` intact).
- Tokenisation (leçon 12) : `train_v2.bin` = 64 301 023 tokens (contre 66 216 164, 2,89 % de moins) ; même BPE, **validation inchangée** (non filtrée) pour rester comparable.
- Réglage pour lancer D : `DONNEES_TRAIN = "train_v2"` et un nouveau `NOM` dans `07_entrainement_corpus.py`, sur la base retenue (A si A améliore B, sinon B).
- **Réglage prêt** (11 novembre 2026) : `DONNEES_TRAIN = "train_v2"`, `NOM = "livres_D_filtre"` ; tout le reste identique à A (d 384, 6 blocs, 6 têtes, T 256, batch 32, 20 000 étapes, LR 6e-4 → 6e-5, graine 0). Base retenue : A.
- **Une seule variable change** : les données d'entraînement (66 216 164 → 64 301 023 tokens, 1,93 % des caractères retirés). BPE et validation (3 046 367 tokens) inchangés : un gain prouve un meilleur apprentissage, pas un test plus facile. D voit un peu moins de tokens, donc un peu plus de passages (environ 2,55).
- **Ce qui est retiré** : code web (`@font-face`, CSS), mentions d'édition et ISBN, codes isolés, lignes presque sans lettres, alphabets rares (136 caractères gardés sur 611).
- **À comparer** : validation de A = 3,068 (perplexité 21), train 2,842, écart 0,226, durée 74 min. Commande : `.venv/bin/python src/06_corpus/scripts/07_entrainement_corpus.py`. Fichiers : `models/checkpoints/livres_D_filtre_{meilleur,dernier}.pt`, `logs/entrainement_livres_D_filtre.csv`.
- Attente honnête : le bruit est faible (moins de 2 %), le gain sera petit ; l'effet visible devrait être l'absence de faux CSS et de faux ISBN dans les textes.

### Expérience D : résultats (11 novembre 2026)

Sortie complète : `logs/entrainement_livres_D_filtre_sortie.txt` ; courbe : `logs/entrainement_livres_D_filtre.csv` (s'arrête à l'étape 19 500 : une manipulation git a détourné le fichier, la fin a été récupérée depuis le processus ; la ligne 20 000 figure dans la sortie).

| Mesure | A (`train`) | D (`train_v2`) |
|---|---|---|
| Tokens d'entraînement | 66 216 164 | 64 301 023 |
| Durée | 74 min | 74 min |
| Validation (non filtrée) | **3,068 (21)** | 3,106 (22) |
| Train | 2,842 | 2,871 |

- Sur la validation d'origine, D est moins bon de 0,038 depuis le début de l'entraînement (écart de 0,03 à 0,04 à chaque étape).
- **Hypothèse** : la validation contient encore du bruit (1,23 % des tokens) que A a appris à prédire et pas D. Test (leçon 14, 800 fenêtres régulières, CPU, checkpoints à 20 000) :

| | Validation non filtrée | Validation filtrée |
|---|---|---|
| A | 3,0556 | 3,0794 |
| D | 3,0931 | 3,0777 |
| D − A | +0,038 | **−0,002** |

- Effet du bruit : −0,024 pour A (le bruit l'avantage), +0,016 pour D (il le pénalise). Somme : 0,040, soit l'écart observé. **Hypothèse confirmée.**
- Conclusion : le filtrage ne donne **pas de gain mesurable** (écart de −0,002, dans le bruit de mesure de 800 fenêtres). Il retire du bruit sans changer la qualité du français. Le train de D est plus haut que celui de A (2,871 contre 2,842) : le code web et les ISBN retirés étaient faciles à prédire.
- Leçon de méthode : un changement sur les données d'entraînement doit s'évaluer sur une validation qui subit **le même changement**, sinon la comparaison est biaisée. La validation de référence (non filtrée) reste utile pour comparer à 0, B et A.
- Base retenue pour la suite : **A** (D n'apporte rien et demande une validation de plus).
- Sorties de la leçon 14 : `logs/validation_filtree_sortie.txt`.

### Expérience A : d = 384, 6 blocs, 6 têtes, T = 256 (11 novembre 2026)

Sortie : `logs/entrainement_livres_A_d384_sortie.txt` ; courbe : `logs/entrainement_livres_A_d384.csv` ; fichiers `livres_A_d384_*`.

| Mesure | Référence | B | A |
|---|---|---|---|
| Paramètres | 5,55 M | 5,59 M | 14,29 M |
| Validation | 3,321 (28) | 3,227 (25) | **3,068 (21)** |
| Train | 3,141 | 3,045 | 2,842 |
| Écart val − train | 0,180 | 0,182 | 0,226 |
| Durée | 14 min | 30 min | 74 min (estimation 71-90 : juste) |
| VRAM entraînement | n. m. | 1 500 MiB | 2 422 MiB |

- Gain de 0,159 par rapport à B (perplexité 25 → 21), plus que le gain de B sur la référence (0,094). La capacité était bien un goulot.
- L'écart val − train grandit (0,226) : le modèle commence à mieux coller à l'entraînement, sans surapprentissage grave (la validation baisse jusqu'au bout).
- La validation baisse encore à l'étape 20 000 (gain de 0,003 par 500 étapes) : plus d'étapes (essai C) aideraient.
- Deux choses ont changé (taille et LR 6e-4) : on ne sépare pas leur part respective.
- Base retenue pour la suite (cumulatif) : A.
