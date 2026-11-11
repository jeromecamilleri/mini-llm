# Leçon 6 : le bloc Transformer

Script : [`06_bloc_transformer.py`](../scripts/06_bloc_transformer.py)
Code : [`tools/transformer_block.py`](../tools/transformer_block.py)
Lancement : `.venv/bin/python src/03_transformer/scripts/06_bloc_transformer.py`

## L'essentiel en 30 secondes

Un **bloc** = deux étapes :

| Étape | Ce qui se passe | Analogie |
|---|---|---|
| 1. Attention | les mots s'échangent des infos | discussion en classe |
| 2. Feed forward | chaque mot traite ce qu'il a reçu, **seul** | réflexion individuelle |

Un GPT = ces blocs **empilés** : la sortie du bloc 1 entre dans le bloc 2, et ainsi de suite.

```
mots -> [bloc 1] -> [bloc 2] -> ... -> [bloc 12] -> prédiction du mot suivant
```

**« 12 blocs » pour GPT-2** veut dire 12 blocs l'un après l'autre. Ils ont la même **structure** mais **pas les mêmes poids** : chaque bloc a ses propres matrices, apprises séparément. Les premiers blocs captent plutôt des liens simples, les suivants des liens plus abstraits.

Deux ingrédients sont des **sécurités**, pas des idées nouvelles :

| Ingrédient | À quoi ça sert | En une phrase |
|---|---|---|
| Connexion résiduelle | ne pas perdre l'information d'origine | on **ajoute** une correction au lieu de tout réécrire |
| LayerNorm | garder des nombres de taille raisonnable | on recadre chaque mot avant de calculer |

Si tu retiens « discussion, puis réflexion seul, empilé 12 fois », tu as l'architecture. Le reste de la fiche donne le détail de chaque ingrédient.

## Pourquoi cette leçon ?

Les leçons 1 à 5 font **parler** les tokens entre eux. Mais l'attention ne fait que **mélanger** des vecteurs : il manque de quoi **réfléchir** sur ce mélange. Le **bloc Transformer** assemble quatre ingrédients :

| Ingrédient | Rôle | Image |
|---|---|---|
| **Attention** (leçons 1 à 5) | les tokens se parlent | la discussion |
| **Feed forward** | petit réseau classique, appliqué à **chaque token séparément** | chacun réfléchit seul |
| **Connexion résiduelle** | on **additionne** le résultat à l'entrée au lieu de la remplacer | on corrige, on ne réécrit pas |
| **LayerNorm** | on remet chaque token à une échelle raisonnable | on recadre |

**Schéma d'un bloc** (variante de GPT-2) :

```
X = X + attention( layer_norm(X) )      # les tokens se parlent
X = X + feed_forward( layer_norm(X) )   # chaque token réfléchit seul
```

Entrée et sortie ont la **même forme** (T, d) : on peut empiler autant de blocs qu'on veut.

## Ingrédient par ingrédient

**LayerNorm** : pour chaque token, moyenne 0 et écart-type 1 sur ses d nombres. Chaque token est normalisé **seul**.

| Entrée | Sortie |
|---|---|
| [1, 2, 3, 4] | [−1.34, −0.45, 0.45, 1.34] |
| [10, 20, 30, 40] | [−1.34, −0.45, 0.45, 1.34] |

Même résultat malgré des échelles très différentes. Dans un réseau profond, les nombres peuvent grossir ou rétrécir à chaque couche ; la normalisation les garde dans une plage où l'entraînement reste stable.

**Feed forward** : deux couches linéaires avec une activation entre les deux, `d → 4d → d` (pour d = 4 : 148 paramètres). L'activation est **GELU**, une ReLU lissée : GELU(−2) = −0,05 ; GELU(0) = 0 ; GELU(2) = 1,95. C'est elle qui remplace le `tanh` de `01_fundamentals`, qui ne servait que pour le XOR.

**Test « chaque token réfléchit seul »** : on change le 1er token et on regarde si les sorties des tokens 2 à 4 bougent.

| Sous-couche | Les autres tokens changent ? |
|---|---|
| Attention | **oui** (les tokens se parlent) |
| Feed forward | **non** (chaque token est traité seul) |

**Connexion résiduelle** : `sortie = entrée + correction`.

| | Valeur |
|---|---|
| Entrée de `dort` | [0, 1, 0, 0] |
| Correction (attention) | [0.06, 0.09, −0.06, 0.06] |
| Entrée + correction | [0.06, 1.09, −0.06, 0.06] |

Si la sous-couche vaut 0, le bloc rend l'entrée **telle quelle** (vérifié).

## Un bloc complet (paramètres au hasard)

| Token | Avant | Après |
|---|---|---|
| le | [0, 0, 1, 0] | [0.22, 0.80, 0.88, 0.21] |
| chat | [1, 0, 0, 0] | [1.79, 0.18, −0.48, 0.56] |
| dort | [0, 1, 0, 0] | [0.52, 1.29, −0.21, −0.79] |
| . | [0, 0, 0, 1] | [0.11, 0.33, −0.86, 0.62] |

Les paramètres sont tirés au hasard : on regarde la **mécanique**, pas un résultat appris. Forme (4, 4) → (4, 4), et 3 blocs empilés donnent encore (4, 4).

## Ordre de grandeur : GPT-2 (d = 768, 12 blocs)

| Élément | Paramètres |
|---|---|
| Attention par bloc (`4 d²`) | 2 359 296 |
| Feed forward par bloc (`8 d²`) | 4 718 592 |
| 12 blocs | 84 934 656 |
| Embeddings (50 257 tokens + 1 024 positions) | 39 383 808 |
| **Total** | **124 318 464** (≈ 124 millions annoncés) |

## Pourquoi 12 blocs ? Et GPT-4 ?

Ce n'est pas une limite technique : le nombre de blocs est un **choix de taille**, qui dépend du matériel disponible. Plus de blocs et plus de dimensions donnent un modèle plus performant, mais plus coûteux (chaque bloc ajoute environ 12·d² paramètres, soit 7 millions pour d = 768).

| Modèle | Blocs | Dimension d | Paramètres |
|---|---|---|---|
| GPT-2 small | **12** | 768 | 124 M |
| GPT-2 medium | 24 | 1024 | 355 M |
| GPT-2 large | 36 | 1280 | 774 M |
| GPT-2 XL | 48 | 1600 | 1,5 Md |
| GPT-3 | 96 | 12288 | 175 Md |
| Llama 3 8B / 70B | 32 / 80 | – | 8 Md / 70 Md |
| GPT-4 et modèles récents d'OpenAI | non publié | non publié | non publié |

Pourquoi ? Le rapport technique de GPT-4 (2023) dit ne pas publier la taille du modèle, les données ni le calcul, à cause du « paysage concurrentiel » et des « enjeux de sécurité ». Les modèles à **poids ouverts** (Llama, Mistral…) publient eux leur architecture : c'est ce qui rend possible de reproduire GPT-2 à petite échelle dans ce projet.

**Notre carte.** Pendant l'entraînement, chaque paramètre demande environ 16 octets (poids, gradient, et deux valeurs d'Adam, `01_fundamentals` leçon 8).

| Modèle | Mémoire estimée | GTX 1070 (8 Go) |
|---|---|---|
| 124 M (GPT-2 small) | ≈ 2 Go + calculs intermédiaires | possible, lent |
| 774 M | ≈ 12 Go | impossible |
| Mini-GPT visé (4 blocs, d = 256) | ≈ 3 M de paramètres dans les blocs, plus les embeddings | très à l'aise |

Pour notre projet, c'est donc le **matériel** qui fixe la taille qu'on peut se permettre, d'où un mini-GPT de quelques blocs.

## Ce qu'on en déduit

- **Attention + feed forward = deux rôles complémentaires** : les tokens échangent, puis chacun traite ce qu'il a reçu.
- **Résiduelle = on part de « ne rien changer »** : le bloc apprend seulement des corrections, et la rétropropagation (`01_fundamentals`, leçon 5) dispose d'un chemin direct qui traverse l'addition sans être affaibli. C'est ce qui permet d'empiler beaucoup de blocs.
- **LayerNorm avant chaque sous-couche** : les nombres restent à une échelle stable.
- **Les deux tiers des paramètres d'un bloc sont dans le feed forward** : c'est là que le modèle mémorise beaucoup de savoir.
- **Même forme en entrée et en sortie** : on peut empiler. Les premiers blocs captent des liens simples, les suivants des liens plus abstraits.
- **On retrouve le « 124 millions » de GPT-2** avec ces seules formules : il n'y a rien de caché.

## Et ensuite ?

Leçon 7 : le **petit GPT complet**. Embeddings de tokens + positions, une pile de blocs, une couche finale qui donne un score par mot du vocabulaire, et une première génération de texte (encore avec des poids au hasard).
