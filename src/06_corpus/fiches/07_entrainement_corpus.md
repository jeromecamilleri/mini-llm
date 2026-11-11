# 07 — Entraîner un GPT sur les livres

Script : [`../scripts/07_entrainement_corpus.py`](../scripts/07_entrainement_corpus.py) · données : [`../tools/donnees_corpus.py`](../tools/donnees_corpus.py)

## L'essentiel en 30 secondes

- Même boucle que `04_training` : tirer un batch, calculer la perte, rétropropager, pas d'AdamW, mesurer la validation, sauver le meilleur modèle.
- Ce qui change, c'est l'échelle : 66 millions de tokens, 4 612 tokens de vocabulaire, environ **5,55 millions de paramètres**, contexte de 128 tokens. Le GPU devient indispensable.
- Trois ajouts pour rester stable : **échauffement** du taux d'apprentissage, **décroissance en cosinus**, **écrêtage des gradients**.

## Ce qui change par rapport aux animaux

| | `04_training` | ici |
|---|---|---|
| corpus | 67 mots | 66 millions de tokens |
| vocabulaire | 13 mots | 4 612 tokens (BPE) |
| paramètres | 26 000 | 5 553 664 |
| contexte T | 4 | 128 |
| batch | 16 fenêtres | 32 fenêtres de 128 tokens |
| validation | 7 mots | 17 livres entiers (3 millions de tokens) |

Dans les 5,55 millions de paramètres, 2,36 millions sont dans les embeddings (4 612 tokens × 256 dimensions, deux fois : entrée et sortie).

## Les trois ajouts

| Ajout | Ce que ça fait | Pourquoi |
|---|---|---|
| échauffement (300 étapes) | le taux part de 0 et monte jusqu'à 1e-3 | au début les poids sont aléatoires, un grand pas ferait n'importe quoi |
| cosinus | le taux redescend vers 1e-4 | grands pas pour dégrossir, petits pas pour affiner |
| écrêtage | gradient ramené à une norme de 1 si plus grand | un seul batch bizarre ne casse plus le modèle |

## Lire la perte

- Un modèle au hasard : ln(4612) = **8,44**.
- **Perplexité** = exp(perte) = parmi combien de tokens le modèle hésite. Perte 5 : 150 tokens ; perte 4 : 55 ; perte 3 : 20.
- Elle ne descend jamais à 0 : après « il », beaucoup de mots sont possibles.
- Test de vérification (40 étapes seulement) : la validation passe de 8,32 à 7,51, et le texte généré est encore du charabia (« Il était une foisig Cor évogar… »). Normal au début.

## Réglages

`N_ETAPES = 20 000` (environ 1,2 passage sur le corpus), `TAILLE_BATCH = 32`, `LR_MAX = 1e-3`, mesure tous les 500 pas. `MAX_MINUTES` fixe un budget de temps ; Ctrl-C arrête proprement ; `REPRENDRE = True` repart du dernier état.

Fichiers : `models/checkpoints/livres_meilleur.pt`, `livres_dernier.pt`, et `logs/entrainement_livres.csv`.

## À observer quand tu lances le script

- La vitesse et le temps restant affichés à chaque mesure.
- La perte de validation par rapport à celle d'entraînement : si l'écart se creuse, c'est du surapprentissage.
- Le texte généré toutes les 500 étapes : charabia, puis mots corrects, puis phrases plausibles.

## Le taux d'apprentissage en cosinus

Le **taux d'apprentissage** est la taille du pas que le modèle fait à chaque étape pour corriger ses poids. Trop grand : il oscille et dépasse la bonne valeur. Trop petit : il avance à peine.

On le fait donc varier selon l'avancement. Image : poser un objet à un endroit précis, on approche vite de la main, puis on ralentit en s'en rapprochant.

| Étape | Taux | Phase |
|---|---|---|
| 0 → 300 | de 0 à 1,0e-3 | échauffement : on monte doucement, car les poids sont encore aléatoires |
| 5 000 | 8,8e-4 | grands pas : le modèle dégrossit |
| 10 000 | 5,6e-4 | milieu : la courbe descend vite |
| 15 000 | 2,4e-4 | petits pas : il affine |
| 20 000 | 1,0e-4 | fin |

La forme est celle d'un cosinus de 0 à π : elle part à plat, descend vite au milieu, finit à plat.

Conséquence : **la perte ralentit vers la fin**, en partie parce que les pas sont plus petits (c'est voulu), en partie parce que les gains faciles sont déjà pris.

## Pourquoi le modèle ralentit et manque de cohérence

La perte passe de 8,4 à 4,3 en 1 000 étapes, puis ne gagne plus que 0,03 à 0,05 par tranche de 500. Le texte n'a pas de fil : chaque phrase est plausible, mais elles ne se suivent pas.

| Limite | Effet |
|---|---|
| contexte de 128 tokens (environ 100 mots) | il oublie le début d'une histoire |
| 5,5 millions de paramètres | trop petit pour retenir des faits ou suivre une intrigue |
| 1,2 passage sur 66 M de tokens | chaque page vue environ une fois |
| pas de but | il prédit le mot suivant, il ne raconte rien |

Pour comparaison : GPT-2 a 124 millions de paramètres, soit 22 fois plus. Les pistes (une variable à la fois) : modèle plus gros, contexte plus long, plus d'étapes, meilleur nettoyage, meilleur tokenizer.

## La taille de batch (TAILLE_BATCH)

C'est le nombre de fenêtres de T tokens lues **avant** chaque correction des poids.

| | Batch petit | Batch grand |
|---|---|---|
| correction | bruitée (peu d'exemples) | plus fiable (moyenne sur beaucoup) |
| mémoire GPU | peu | beaucoup |
| tokens lus par étape | TAILLE_BATCH × T, donc moins | donc plus |

Ici 32 × 128 = 4 096 tokens par étape. Pour l'expérience T = 256, on garde 32 × 256 = 8 192 tokens par étape : le batch reste celui de la référence, mais le modèle voit deux fois plus de données (2,5 passages), ce qui se mélange à l'effet du contexte. L'autre choix, 16 × 256 = 4 096, aurait gardé la même quantité de données.

## Ce qu'on en déduit

- Un GPT de 5 millions de paramètres sur 66 millions de tokens ne sera pas un « ChatGPT » : on attend des phrases souvent cohérentes localement, mais sans suite dans les idées.
- On change **une seule variable** à la fois quand on compare (taille du modèle, nombre d'étapes, tokenizer, corpus).
- Le meilleur checkpoint se choisit sur la validation (livres jamais vus), pas sur l'entraînement.
