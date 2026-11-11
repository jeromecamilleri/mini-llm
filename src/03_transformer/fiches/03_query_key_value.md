# Leçon 3 : Query, Key, Value

Script : [`03_query_key_value.py`](../scripts/03_query_key_value.py)
Code : [`tools/attention.py`](../tools/attention.py) (`attention_qkv`)
Lancement : `.venv/bin/python src/03_transformer/scripts/03_query_key_value.py`

## Pourquoi cette leçon ?

**Rappel des leçons 1 et 2.** `E` est le tableau des vecteurs, un par token (une ligne chacun). `E @ E.T` multiplie chaque ligne par chaque ligne : c'est le produit scalaire de **toutes les paires** de tokens, en une seule opération. On obtient un tableau 4×4 de scores.

Exemple : `dort` avec `chat` = 0,5×2,0 + 2,0×0,5 = **2,0**.

**Le défaut.** On compare chaque vecteur **tel quel**, avec une seule définition de la ressemblance. Or un vecteur ressemble toujours beaucoup à lui-même : `dort` avec `dort` = 0,5×0,5 + 2,0×2,0 = **4,25**, le plus grand score de la ligne. D'où les 89 % de `dort` sur lui-même (leçon 2). Pourtant `dort` veut surtout savoir **qui** dort : regarder `chat`.

**La solution.** Avant de comparer, on transforme chaque token avec trois petites matrices, qui seront **apprises**. Chaque token a alors **trois versions** :

| Nom | Rôle | Question posée |
|---|---|---|
| **Query (Q)** | la question | « que cherche-je chez les autres ? » |
| **Key (K)** | l'étiquette | « que présente-je pour être trouvé ? » |
| **Value (V)** | le contenu | « qu'apporté-je si on me choisit ? » |

**Image : une bibliothèque.** Tu arrives avec une question (Q). Chaque livre a une étiquette sur sa tranche (K). Plus ta question correspond à l'étiquette, plus tu ouvres le livre. Ce que tu lis dedans, c'est son contenu (V), pas forcément l'étiquette.

## Les formules

| Étape | Calcul |
|---|---|
| Trois versions | `Q = E @ Wq`, `K = E @ Wk`, `V = E @ Wv` |
| Scores | `Q @ K.T / racine(dk)`, puis masque causal (leçon 2) |
| Poids | softmax (leçon 1) |
| Sortie | `poids @ V` |

`Wq`, `Wk`, `Wv` sont des paramètres comme `W1` de `01_fundamentals` : on les corrige par rétropropagation. Ici, on les écrit **à la main** pour voir l'effet.

## Attention contre neurone classique

(Les « questions » et la « bibliothèque » sont des **images** : personne ne pose réellement de question, voir « Qui pose la question ? » dans la [leçon 5](05_plusieurs_tetes.md). Ce sont des matrices apprises.)

La différence essentielle : **un neurone classique traite chaque token seul, l'attention fait parler les tokens entre eux.**

| | Neurone / couche classique | Attention |
|---|---|---|
| Ce qu'elle fait | transforme **un** token | **mélange** les tokens entre eux |
| Voit les voisins ? | non | oui |
| Ce qui est appris | les poids `W` | `Wq`, `Wk`, `Wv` (aussi des poids) |
| Comment elle combine | par des poids fixes : `x @ W` | par des pourcentages **recalculés pour chaque phrase** |

**Analogie** : une classe de 4 élèves (le, chat, dort, .). Une couche classique, c'est chaque élève qui réfléchit **seul** sur sa feuille. L'attention, c'est une **discussion** où chacun écoute les autres, plus ou moins selon ce qu'ils disent.

Les trois premières lignes du tableau des formules (`E @ Wq`, `E @ Wk`, `E @ Wv`) sont des couches linéaires comme `X @ W1` de `01_fundamentals`. Les trois suivantes (scores, softmax, mélange) n'ont **aucun paramètre** : elles multiplient des résultats qui dépendent de la phrase.

Dans un GPT, les deux se complètent (leçon 6) : l'attention (les tokens se parlent), puis un petit réseau classique appelé **feed forward** (chaque token réfléchit seul sur ce qu'il a appris). On répète ce bloc plusieurs fois.

## Résultats sur « le chat dort . » (dimension 2)

Matrices choisies : `Wq` échange les deux axes (« je cherche l'autre axe »), `Wk` = identité, `Wv` réduit l'axe 2 de moitié.

| Token | E | Q | K | V |
|---|---|---|---|---|
| le | [0.5, 0] | [0, 0.5] | [0.5, 0] | [0.5, 0] |
| chat | [2, 0.5] | [0.5, 2] | [2, 0.5] | [2, 0.25] |
| dort | [0.5, 2] | [2, 0.5] | [0.5, 2] | [0.5, 1] |
| . | [0, 0.5] | [0.5, 0] | [0, 0.5] | [0, 0.25] |

La question de `dort` (`[2, 0.5]`) ressemble à l'étiquette de `chat` (`[2, 0.5]`) : un verbe cherche un animal.

Score `dort → chat` : Q(dort)·K(chat) = 2×2 + 0,5×0,5 = 4,25, divisé par √2 = **3,01**.

**Poids après masque et softmax :**

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 100 % | 0 % | 0 % | 0 % |
| chat | 22 % | 78 % | 0 % | 0 % |
| dort | 8 % | 77 % | 16 % | 0 % |
| . | 22 % | 37 % | 22 % | 18 % |

Ligne `dort` :

| | le | chat | dort |
|---|---|---|---|
| Leçon 2 (`E @ E.T`) | 2 % | 9 % | **89 %** |
| Avec Q/K/V | 8 % | **77 %** | 16 % |

Nouveau vecteur de `dort` = 8 % V(le) + 77 % V(chat) + 16 % V(dort) = `[1.65, 0.35]`.

## Ce qu'on en déduit

- **`dort` regarde maintenant `chat`** (77 % au lieu de 9 %) : il va chercher **qui** dort. C'est ce que `E @ E.T` ne pouvait pas faire.
- **Comparer et transmettre sont séparés** : Q et K décident **qui regarder**, V décide **ce qu'on récupère**. Un token peut être trouvé grâce à son étiquette et apporter autre chose.
- **Pourquoi diviser par √dk** : un produit scalaire grandit avec la dimension. Avec 768 nombres, les scores seraient énormes et le softmax mettrait ~100 % sur un seul token. On les ramène à une taille raisonnable.
- **Ici 12 nombres écrits à la main** (3 matrices 2×2). Dans le GPT, ils sont tirés au hasard puis **appris** : le modèle découvre tout seul quelles questions poser. Avec d = 768 (GPT-2), c'est 1 769 472 nombres par tête.
- **Rien d'autre ne change** : masque, softmax et mélange sont ceux des leçons 1 et 2.

## Et ensuite ?

Leçon 4 : les **positions**. Pour l'instant, `le chat dort` et `dort chat le` donneraient les mêmes vecteurs : le modèle ne connaît pas l'ordre des mots.
