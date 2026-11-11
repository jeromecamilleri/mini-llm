# Exercices : entraîner le réseau sur le XOR

Cours associé : [`../../scripts/a_la_main/06_entrainement.py`](../../scripts/a_la_main/06_entrainement.py) (module : [`../../tools/backpropagation.py`](../../tools/backpropagation.py))

Même exemple que les fiches 03 à 05 : XOR, réseau `[2, 4, 1]` (`tanh` puis `sigmoid`), perte MSE.

## Le principe

Une **époque** = un passage sur les 4 exemples :

1. propagation avant, puis perte (fiche 03)
2. rétropropagation : gradients (fiche 05)
3. descente de gradient : `poids = poids − taux × gradient`

On répète jusqu'à un **critère d'arrêt**. Ici : taux = 0.5, arrêt quand la perte passe sous 0.01 (garde-fou : 20 000 époques).

## Exemple résolu : les poids bougent

| époque | perte | `W1[0,0]` | `W1[1,0]` | `W2[0,0]` | `b2` |
|--------|-------|-----------|-----------|-----------|------|
| 0 | 0.2687 | 0.126 | −0.536 | −0.498 | 0.000 |
| 10 | 0.2565 | 0.084 | −0.547 | −0.517 | 0.055 |
| 50 | 0.2428 | 0.015 | −0.511 | −0.492 | −0.009 |
| 100 | 0.2150 | 0.061 | −0.444 | −0.433 | −0.091 |
| 200 | 0.1028 | 1.086 | −0.491 | −1.164 | −0.099 |
| 300 | 0.0249 | 1.981 | −0.949 | −2.244 | 0.021 |
| **418** | **0.0100** | 2.328 | −1.180 | −2.747 | 0.121 |

À lire :
- **Début lent** : pendant ~100 époques, la perte reste proche de 0.25 (le hasard) et les poids bougent à peine. Le réseau cherche d'abord une direction utile.
- **Chute rapide** entre les époques 100 et 300, puis ralentissement : les gradients diminuent à mesure que l'on s'approche du but.
- **Arrêt à l'époque 418** : perte < 0.01.

## Utiliser le réseau

On n'a plus besoin de la rétropropagation, seule la propagation avant sert.

| entrée | sortie | réponse (seuil 0.5) | attendu |
|--------|--------|---------------------|---------|
| (0, 0) | 0.055 | 0 | 0 ✓ |
| (0, 1) | 0.892 | 1 | 1 ✓ |
| (1, 0) | 0.902 | 1 | 1 ✓ |
| (1, 1) | 0.125 | 0 | 0 ✓ |

Pour des entrées intermédiaires jamais vues : `(0.9, 0.1)` → 0.829 (réponse 1), `(0.1, 0.1)` → 0.097 (réponse 0), `(0.5, 0.5)` → 0.216 (réponse 0). Le réseau n'a vu que 4 cas : ailleurs, il extrapole à sa façon.

## Un modèle « à poids ouverts », c'est quoi ?

Un modèle « à poids ouverts » (Llama, Mistral, GPT-2…), c'est :

1. **L'architecture** : la forme du réseau (ici `[2, 4, 1]` avec `tanh` puis `sigmoid`).
2. **Les poids** : tous les nombres appris. Chez toi, ce sont les 17 (`W1` 8, `b1` 4, `W2` 4, `b2` 1).

Avec les deux, n'importe qui peut refaire la propagation avant et utiliser le modèle, sans refaire l'entraînement.

**Ce qui n'est pas dans les poids**

- Les **données d'entraînement** : souvent non publiées (d'où « poids ouverts » plutôt que « open source » complet).
- Le **code d'entraînement** et les réglages (taux, nombre d'étapes) : parfois publiés, parfois non.

**Les ordres de grandeur**

- Ton réseau : 17 paramètres, quelques centaines d'octets.
- GPT-2 : 124 millions, environ 500 Mo.
- Llama 3 : de 8 à 70 milliards, de 16 à 140 Go.

Un fichier de poids est donc simplement de gros tableaux de nombres (`W1`, `b1`…) sauvegardés, par exemple en `.safetensors`.

## Ce qu'on en déduit

- **Quand s'arrêter ?** Trois critères courants : perte sous un seuil, nombre maximal d'époques (garde-fou), perte qui stagne. Sans critère, la perte baisse de moins en moins et n'atteint jamais exactement 0.
- **Le taux d'apprentissage est un compromis.** Trop petit : l'entraînement est très lent. Trop grand : on « saute » par-dessus le creux et la perte peut osciller ou diverger. Avec 1.0, le même réseau atteint le seuil en ~210 époques au lieu de ~420.
- **La sortie n'est jamais exactement 0 ou 1** : on la transforme en réponse avec un seuil (0.5). Un `0.89` est « sûr à 89 % », pas une certitude.
- **Pourquoi le réseau marche sur les 4 cas mais pas forcément ailleurs** : il a appris ces 4 exemples, pas la règle. Sur un vrai modèle on mesure donc la perte aussi sur des **données non vues à l'entraînement** (sur-apprentissage, cours à venir).
- **Dépend du départ** : les poids initiaux sont aléatoires (graine `seed=0`). Une autre graine donne un autre parcours, mais ici toutes celles testées (0 à 5) convergent, entre ~270 et ~420 époques.
- **Lien avec le GPT** : l'entraînement d'un LLM est cette même boucle (perte, rétropropagation, descente), répétée des millions de fois sur de très gros jeux de données, avec une perte différente (entropie croisée) et un optimiseur plus sophistiqué (Adam). Les critères d'arrêt (seuil, nombre d'étapes, perte de validation) sont les mêmes idées.
- **Expérience à faire plus tard** (règle du projet : une seule variable à la fois) : même réseau, activation cachée différente (relu, leaky_relu, sigmoid), et comparer le nombre d'époques nécessaires.

## Réflexion : un `if` ou un réseau entraîné ?

Pour le XOR, on aurait pu écrire à la main :

```python
def xor(x1, x2):
    if x1 != x2:
        return 1
    return 0
```

Ici, on a fait autre chose : **on n'a écrit aucune règle**. On a donné 4 exemples (entrées et réponses attendues) à un réseau de 4 neurones, et il a **trouvé tout seul** les 17 nombres (poids et biais) qui reproduisent le XOR.

|  | `if` écrit à la main | réseau entraîné |
|--|----------------------|-----------------|
| la règle vient de | le programmeur | les **données** |
| il faut connaître la règle | oui | non |
| entrées hors des cas prévus | comportement à définir | extrapole (sortie continue) |
| coût | une ligne | ~420 époques de calcul |

Pour le XOR, le `if` est évidemment plus simple, et c'est voulu : le XOR est un exemple **minuscule dont on connaît la réponse**, ce qui permet de vérifier que le réseau a bien appris.

Le vrai intérêt apparaît quand **personne ne sait écrire la règle** : reconnaître un chat sur une photo, traduire une phrase, prédire le mot suivant. Aucun `if` imbriqué n'y suffit. On ne programme plus le comportement, on programme **comment apprendre** (propagation avant, perte, rétropropagation, descente), puis on laisse les données façonner les poids.

C'est exactement le principe de ce projet : un GPT n'a aucune règle de grammaire écrite à la main. Il a la même boucle que celle-ci, avec des milliards de poids et un corpus de textes à la place des 4 exemples du XOR. D'où la question centrale du projet : **comment le corpus d'entraînement change-t-il ce que le modèle sait faire ?**
