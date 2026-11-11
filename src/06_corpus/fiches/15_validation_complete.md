# 15 — Mesurer la perte sur toute la validation

Script : [`../scripts/15_validation_complete.py`](../scripts/15_validation_complete.py)

## L'essentiel en 30 secondes

- La perte de validation affichée pendant l'entraînement est une **mesure rapide** : 40 batchs de 32 fenêtres tirés au hasard, soit environ 11 % de la validation.
- Cette leçon la mesure **sur 100 % de la validation**, du début à la fin, avec une fenêtre qui glisse : chaque token est compté une fois.
- Elle donne aussi les **bits par octet**, une unité qui ne dépend pas du tokenizer et qui se compare à un autre modèle.
- Un écart de moins de 0,02 entre deux essais n'est fiable qu'avec cette mesure.

## Les quatre faiblesses de la mesure rapide

| # | Faiblesse | Effet |
|---|---|---|
| 1 | Seulement 11 % de la validation | un écart de 0,01 peut venir du choix des fenêtres |
| 2 | Fenêtres tirées au hasard | elles peuvent se chevaucher, des zones ne sont jamais lues |
| 3 | Début de fenêtre sans contexte | le 2e token est prédit avec 1 token de contexte, le 256e avec 255 : les premiers sont mal prédits, la perte est un peu **gonflée** |
| 4 | T = 128 pour la référence, 256 pour les autres | le contexte n'est pas le même : comparaison approximative |

## La fenêtre glissante

Exemple avec T = 8 et un pas de 4. Les cases comptées sont en gras.

```
texte :     t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 ...
fenêtre 1 : [t1 t2 t3 t4 t5 t6 t7 t8]            tout est compté
fenêtre 2 :             [t5 t6 t7 t8 t9 t10 t11 t12]   seuls t9 à t12 sont comptés
fenêtre 3 :                         [t9 ... t16]       seuls t13 à t16 sont comptés
```

Chaque token est compté **une seule fois**, et il a toujours au moins la moitié de la fenêtre (T/2 tokens) comme contexte. Le prix : on lit le texte deux fois, donc c'est deux fois plus long.

## Lancer

```bash
.venv/bin/python src/06_corpus/scripts/15_validation_complete.py                  # CPU (défaut) : B, A, D
.venv/bin/python src/06_corpus/scripts/15_validation_complete.py cuda             # GPU
.venv/bin/python src/06_corpus/scripts/15_validation_complete.py --test           # essai rapide, 50 fenêtres
.venv/bin/python src/06_corpus/scripts/15_validation_complete.py livres_P4_8blocs_meilleur.pt   # un modèle précis
```

- **CPU par défaut** : si un entraînement tourne sur le GPU, les deux se ralentiraient. Durée extrapolée d'un essai : ~10 min pour B, 15 à 20 min pour A, D ou P4 (non mesuré sur la validation entière).
- `--test` ne lit que le début de la validation : ses chiffres ne sont **pas représentatifs**, il sert à vérifier que tout marche. CPU et GPU donnent la même valeur (2,6597 pour A en test).
- Pour voir l'effet de la faiblesse 3, mettre `AUSSI_SANS_CONTEXTE = True` dans le script : il refait la mesure avec des fenêtres disjointes qui comptent leur début.

## Lire le résultat

Le bilan met côte à côte, pour chaque modèle :
- **mesure rapide** : celle de l'entraînement (3,227 pour B, 3,068 pour A) ;
- **complète** : toute la validation avec contexte ;
- **écart** : attendu **négatif**, car le contexte aide ; il doit être à peu près le même pour tous les modèles, sinon le classement change ;
- **bit/octet** : perte × tokens par octet / ln 2. Valeurs de la mesure rapide : référence 1,46, B 1,42, A 1,35.

## Ce qu'on en déduit

- Si les modèles gardent le même **classement** sur la mesure complète, nos conclusions tiennent. Les écarts de plus de 0,05 (largeur, profondeur) sont de toute façon robustes.
- Si l'écart « rapide − complète » varie d'un modèle à l'autre, un des classements de petits écarts (P4 contre A, D contre A) est à revoir.
- Les valeurs absolues de la mesure complète sont **plus basses** que celles de l'entraînement : ne jamais les comparer aux anciennes sans le dire. Toujours comparer deux modèles mesurés de la même façon.
- Cette mesure sert d'arbitre final entre des essais proches ; elle ne remplace pas la lecture du texte généré.
- Il manque toujours un vrai **jeu de test** ouvert une seule fois à la fin : ces mêmes 17 livres servent à choisir et à mesurer.

## Mesurer sur d'autres textes : `--validation=`

Le script accepte `--validation=NOM`, un fichier de `datasets/tokenized/`. Avec `--validation=theatre_validation`, il mesure les modèles sur les **dialogues réservés** à la leçon 17 (277 026 tokens, jamais appris). C'est la **référence à relever avant** d'ajouter du théâtre à l'apprentissage : sans elle, la perte du modèle « avec théâtre » sur les dialogues ne se compare à rien.

Les bits par octet de cette validation sont calculés sur le texte décodé : approximatifs si des tokens inconnus sont perdus (0,08 % dans le théâtre).
