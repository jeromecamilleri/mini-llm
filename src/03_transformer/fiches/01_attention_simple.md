# Leçon 1 : l'attention la plus simple

Script : [`01_attention_simple.py`](../scripts/01_attention_simple.py)
Code : [`tools/attention.py`](../tools/attention.py)
Lancement : `.venv/bin/python src/03_transformer/scripts/01_attention_simple.py`

## Pourquoi cette leçon ?

Un embedding (`02_tokenizer`, leçon 4) donne à un mot **le même vecteur partout** : `dort` est identique dans « le chat dort » et « le moteur dort ». Or le sens d'un mot dépend de ses voisins.

L'**attention** règle ça : chaque token regarde les autres tokens de la phrase et **mélange leurs vecteurs avec le sien**, plus ou moins fort selon leur intérêt.

## Les 3 gestes

| Geste | Opération | Résultat |
|---|---|---|
| 1. **Scores** | produit scalaire de chaque paire : `E @ E.T` | « à quel point se ressemblent-ils ? » |
| 2. **Poids** | softmax de chaque ligne | des pourcentages qui font 100 % |
| 3. **Mélange** | `poids @ E` | nouveau vecteur = moyenne pondérée |

Deux mots nouveaux :
- **Produit scalaire** de a et b = `a1×b1 + a2×b2 + …` : grand si les vecteurs pointent dans la même direction (comme le neurone de `01_fundamentals`).
- **Softmax** : met des scores en pourcentages (`e^score / somme des e^score`). Plus un score est grand, plus son pourcentage l'est.

## Exemple : « le chat dort . » (vecteurs écrits à la main, dimension 2)

| Token | Vecteur | Idée |
|---|---|---|
| le | [0.5, 0.0] | mot outil |
| chat | [2.0, 0.5] | surtout « animal » |
| dort | [0.5, 2.0] | surtout « action » |
| . | [0.0, 0.5] | ponctuation |

**Scores** (ligne = qui regarde, colonne = qui est regardé). Exemple : dort·chat = 0,5×2,0 + 2,0×0,5 = 2,0.

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 0.25 | 1.00 | 0.25 | 0.00 |
| chat | 1.00 | 4.25 | 2.00 | 0.25 |
| dort | 0.25 | 2.00 | 4.25 | 1.00 |
| . | 0.00 | 0.25 | 1.00 | 0.25 |

**Poids** (softmax de chaque ligne) :

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 20 % | 43 % | 20 % | 16 % |
| chat | 3 % | 86 % | 9 % | 2 % |
| dort | 2 % | 9 % | 86 % | 3 % |
| . | 16 % | 20 % | 43 % | 20 % |

**Mélange pour `dort`** : 2 % le + 9 % chat + 86 % dort + 3 % « . » donne `[0.62, 1.78]` au lieu de `[0.5, 2.0]`. Le vecteur a glissé un peu vers `chat` : il porte maintenant un peu de « animal ».

**Softmax sur des scores écartés** : `[1, 1, 1]` → 33 % chacun ; `[1, 2, 3]` → 9 / 24 / 67 % ; `[1, 2, 10]` → presque 100 % pour le dernier.

## Ce qu'on en déduit

- **Le contexte entre dans le vecteur** : un `dort` après `chat` n'a plus le même vecteur qu'un `dort` après `moteur`.
- **Un token se regarde surtout lui-même** (86 %) : le produit scalaire d'un vecteur avec lui-même est toujours grand. L'effet est donc faible. Les leçons 2 et 3 corrigent cela (des vecteurs « Query » et « Key » appris).
- **Entrée et sortie ont la même forme** (4, 2) : on pourra empiler plusieurs blocs.
- **Aucun paramètre appris ici** : tout est calculé à partir de `E`. Rien à entraîner pour l'instant.
- **Défaut pour un GPT** : ici `le` regarde aussi `chat`, qui vient **après** lui. Or un GPT prédit le mot suivant, il ne doit pas voir le futur. C'est le sujet de la leçon 2.

## Et ensuite ?

Leçon 2 : le **masque causal**, pour qu'un token ne regarde que ceux qui le précèdent.
