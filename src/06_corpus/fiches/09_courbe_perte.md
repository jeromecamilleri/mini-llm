# 09 — La courbe d'apprentissage

Script : [`../scripts/09_courbe_perte.py`](../scripts/09_courbe_perte.py)

## L'essentiel en 30 secondes

- On relit `logs/entrainement_livres.csv` et on trace la perte (train et validation) et le taux d'apprentissage dans `logs/courbe_entrainement_livres.png`.
- La courbe montre ce qu'on ressentait à l'écran : le modèle progresse vite, puis de plus en plus lentement.

## Les chiffres du premier entraînement

| Mesure | Valeur |
|---|---|
| validation, étape 500 | 4,794 (perplexité 121) |
| validation, étape 20 000 | 3,321 (perplexité 28) |
| gain par tranche de 500 étapes, au début | 0,519 |
| gain par tranche de 500 étapes, à la fin | 0,001 |
| écart validation − train, à la fin | 0,180 |

## Comment lire la courbe

- **Rendements décroissants** : au début, 500 étapes font gagner 0,5 de perte ; à la fin, 0,001. C'est ce que tu avais remarqué (« ça ralentit »). Une cause est le taux qui descend en cosinus, l'autre est que les gains faciles (orthographe, dialogues) sont pris les premiers.
- **Pas de surapprentissage** : la validation ne remonte pas. L'écart de 0,18 avec le train est faible.
- **Écart faible = modèle limité par sa taille** (5,5 M de paramètres, contexte de 128), pas par le manque de données. Plus de livres ne suffirait pas : un modèle plus gros pourrait mieux les exploiter.

## Ce qu'on en déduit

- Continuer 20 000 étapes de plus avec le même modèle gagnerait peu : la courbe est presque plate.
- La prochaine expérience utile est un changement de **taille** (d, blocs) ou de **contexte T**, une variable à la fois, avec les mêmes réglages de génération pour comparer.
