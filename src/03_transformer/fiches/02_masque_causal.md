# Leçon 2 : le masque causal

Script : [`02_masque_causal.py`](../scripts/02_masque_causal.py)
Code : [`tools/attention.py`](../tools/attention.py) (`masque_causal`, `attention_causale`)
Lancement : `.venv/bin/python src/03_transformer/scripts/02_masque_causal.py`

## Pourquoi cette leçon ?

**Le problème.** On entraîne le modèle à deviner le mot suivant. Pendant l'entraînement, la phrase complète est dans la fenêtre. Sans précaution, pour deviner `dort` il pourrait lire `dort` juste à côté : comme un examen où la correction est visible, il recopie au lieu d'apprendre. C'est la **triche**.

**Le masque = cacher le futur avec la main.** Chaque position ne voit que ce qui la précède :

| Position | Le modèle voit | Il doit deviner | Caché |
|---|---|---|---|
| 1 | le | chat | chat dort . |
| 2 | le chat | dort | dort . |
| 3 | le chat dort | . | . |

Dans le tableau d'attention de la leçon 1, chaque mot donnait un pourcentage à **tous** les autres. Le masque force à **0 %** les mots qui viennent après : ce sont les cases en haut à droite du tableau.

**Pourquoi −∞ ?** On ne peut pas « supprimer » une case du calcul. On lui met un score de −∞ avant le softmax, qui la transforme en exactement 0 %.

**Pourquoi coder un masque, alors qu'on « programme » le modèle ?** Dans un réseau de neurones, on ne programme pas la règle, on la fait **apprendre** : on code la structure, et le réseau cherche ce qui fait baisser la perte. S'il a une porte ouverte vers la réponse, il la prend, car c'est le chemin le plus facile. Le masque est le code qui ferme cette porte.

Sans masque, à l'entraînement :
1. La phrase complète est dans la fenêtre (le plus rapide pour le GPU).
2. Pour prédire `dort`, le token `dort` est accessible via l'attention.
3. Le réseau découvre le raccourci « recopie le mot suivant » : la perte tombe presque à 0.
4. À la **génération**, le mot suivant n'existe pas encore : le raccourci ne marche plus, et rien n'a été appris sur le langage.

| | Entraînement | Génération |
|---|---|---|
| Sans masque | le futur est visible, il copie | le futur n'existe pas, il est perdu |
| Avec masque | il ne voit que le passé | il ne voit que le passé : ça marche |

Le masque garantit que **l'entraînement ressemble à l'usage réel**.

## Comment ça marche, en détail

1. Le masque est une matrice triangulaire : la ligne `i` autorise les colonnes `0` à `i`.
2. On remplace les scores interdits par **−∞** **avant** le softmax.
3. Comme e^(−∞) = 0, leur poids devient exactement **0 %**.

| Masque | le | chat | dort | . |
|---|---|---|---|---|
| le | 1 | . | . | . |
| chat | 1 | 1 | . | . |
| dort | 1 | 1 | 1 | . |
| . | 1 | 1 | 1 | 1 |

(1 = permis, `.` = interdit)

## Résultats sur « le chat dort . »

Poids après masque (comparer à la leçon 1) :

| | le | chat | dort | . |
|---|---|---|---|---|
| le | 100 % | 0 % | 0 % | 0 % |
| chat | 4 % | 96 % | 0 % | 0 % |
| dort | 2 % | 9 % | 89 % | 0 % |
| . | 16 % | 20 % | 43 % | 20 % |

Pour `dort` : sans masque 2 / 9 / 86 / **3 %** (le futur `.` pesait 3 %) ; avec masque 2 / 9 / 89 / **0 %**.

**Test de triche** : on remplace le dernier token par un vecteur très différent et on regarde si les sorties de `le`, `chat`, `dort` changent.

| | Sorties du passé changées ? |
|---|---|
| Sans masque | **oui** (le futur fuit dans le passé) |
| Avec masque | **non** (le passé ignore le futur) |

## Ce qu'on en déduit

- **Le passé ne dépend pas du futur** : c'est vérifié par le test ci-dessus, pas seulement affirmé.
- **Le premier token n'a qu'un choix** : lui-même, à 100 %.
- **Le masque fabrique les T exercices d'un coup** : « le » → chat, « le chat » → dort, etc. (`02_tokenizer`, leçon 5), en **une seule opération**, sans jamais montrer la réponse.
- **Pourquoi −∞ et pas 0** : un score de 0 donnerait encore un poids positif après softmax. −∞ donne exactement 0.
- **Pas de paramètre appris** : le masque est fixe, il ne s'entraîne pas.

## Et ensuite ?

Leçon 3 : **Query, Key, Value**. Les vecteurs qui servent à comparer et ceux qu'on mélange deviennent appris et distincts, ce qui corrige le défaut « un token se regarde surtout lui-même ».
