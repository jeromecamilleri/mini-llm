# Leçon 5 : plusieurs têtes d'attention

Script : [`05_plusieurs_tetes.py`](../scripts/05_plusieurs_tetes.py)
Code : [`tools/attention.py`](../tools/attention.py) (`attention_multi_tetes`)
Lancement : `.venv/bin/python src/03_transformer/scripts/05_plusieurs_tetes.py`

## Pourquoi cette leçon ?

Une attention (leçon 3) pose **une seule question** par token. Mais dans « le chat dort » :

| Mot | Sa question |
|---|---|
| `dort` | **qui** dort ? (→ `chat`) |
| `chat` | quel **article** m'accompagne ? (→ `le`) |

Deux questions différentes, une seule attention : elle doit choisir.

## L'idée

Une **tête** = une attention complète, avec ses propres `Wq`, `Wk`, `Wv`. On en lance **plusieurs en parallèle** :

1. chaque tête pose sa propre question et calcule ses propres poids ;
2. on met leurs résultats **bout à bout** (concaténation) ;
3. une dernière matrice `Wo` les mélange.

Les têtes **ne se parlent pas** : chacune se spécialise pendant l'entraînement.

## Qui pose la question « qui dort ? »

**Personne.** « Qui dort ? » est une **image** pour expliquer. En réalité, la « question » est un **vecteur de nombres** : le token passe dans `Wq`, ce qui donne son Q ; on compare Q aux K des autres (produit scalaire) ; si Q de `dort` ressemble à K de `chat`, l'attention est forte.

| Moment | Contenu de `Wq`, `Wk` |
|---|---|
| Au départ | nombres au hasard, aucune « question » |
| Pendant l'entraînement | corrigés pour améliorer la prédiction du mot suivant |
| À la fin | certains jeux de nombres se comportent comme « le verbe cherche son sujet » |

Dans cette leçon, **nous avons écrit les matrices à la main** pour rendre l'effet lisible. Le vrai modèle ne reçoit aucune consigne : si regarder le sujet aide à prédire le mot suivant, l'entraînement trouve tout seul des matrices qui le font. On interprète ensuite ce qu'elles font.

## Exemple : dimension 4, deux têtes de dimension 2

Traits : `[animal, action, article, ponctuation]`. Matrices écrites à la main pour voir la spécialisation ; dans un vrai GPT elles sont apprises et les rôles sont moins nets.

| Tête | Rôle | Q s'allume pour | K s'allume pour |
|---|---|---|---|
| A | « qui fait l'action ? » | action (verbe) | animal |
| B | « quel article ? » | animal (nom) | article |

**Poids de la tête A** :

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 100 % | 0 % | 0 % | 0 % |
| chat | 50 % | 50 % | 0 % | 0 % |
| dort | 10 % | **81 %** | 10 % | 0 % |
| . | 25 % | 25 % | 25 % | 25 % |

**Poids de la tête B** :

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 100 % | 0 % | 0 % | 0 % |
| chat | **89 %** | 11 % | 0 % | 0 % |
| dort | 33 % | 33 % | 33 % | 0 % |
| . | 25 % | 25 % | 25 % | 25 % |

Quand une tête n'a pas de question pour un token (par exemple la tête B pour un verbe), elle répartit **à égalité**.

**Sortie de `dort`** : tête A `[0.81, 0.10]` (animal, action) + tête B `[0.33, 0]` (article, ponctuation), bout à bout, puis `@ Wo` : `[0.81, 0.10, 0.33, 0]`.

**Vecteurs avant et après** `[animal, action, article, ponct.]` :

| Mot | Avant | Après |
|---|---|---|
| le | [0, 0, 1, 0] | [0, 0, 1, 0] |
| chat | [1, 0, 0, 0] | [0.5, 0, **0.89**, 0] |
| dort | [0, 1, 0, 0] | [**0.81**, 0.10, 0.33, 0] |

`dort` sait maintenant **qui** dort, et `chat` sait qu'il est précédé de `le`.

## Ce qu'on en déduit

- **Plusieurs questions à la fois** : chaque tête lit la même phrase à sa façon.
- **Même coût qu'une grosse tête** : 1 tête de dimension 4 = 3×4×4 = 48 paramètres ; 2 têtes de dimension 2 = 2×3×4×2 = 48. On **découpe** la dimension, on ne l'augmente pas. GPT-2 : 12 têtes × 64 = 768.
- **Entrée et sortie gardent la même forme** (T, d) : on peut empiler les blocs.
- **Les rôles sont découverts, pas imposés** : ici nous les avons écrits ; dans le GPT, l'entraînement trouve les spécialisations utiles (certaines têtes suivent le mot précédent, d'autres la ponctuation…).
- **`Wo`** est une matrice apprise qui recombine les têtes (ici l'identité, pour ne rien cacher).

## Et ensuite ?

Leçon 6 : le **bloc Transformer**. On assemble l'attention (les tokens se parlent) avec un petit réseau classique (chaque token réfléchit seul), des connexions résiduelles et une normalisation.
