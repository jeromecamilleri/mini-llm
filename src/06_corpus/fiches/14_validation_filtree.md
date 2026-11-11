# 14 — A et D sur une validation filtrée

Script : [`../scripts/14_validation_filtree.py`](../scripts/14_validation_filtree.py)

## L'essentiel en 30 secondes

- Pour l'expérience D, le texte d'entraînement était filtré, mais **pas la validation**. D semblait moins bon que A de 0,038.
- Hypothèse : la validation contient encore du bruit (code web, ISBN, caractères rares) que A a appris à prédire et pas D.
- Test : mesurer A et D sur deux validations, l'une non filtrée, l'autre filtrée. Sur la validation filtrée, l'écart est de **−0,002** : à égalité.
- Le filtrage n'apporte donc **pas de gain mesurable** ; l'écart apparent venait du test, pas du modèle.

## Pourquoi quelques tokens suffisent

La perte de validation est une **moyenne** sur tous les tokens. Le bruit ne représente que 1,23 % des tokens de validation, mais :

| Modèle | Perte sur un token de bruit | Pourquoi |
|---|---|---|
| A | presque 0 | il a lu des milliers de lignes `@font-face` |
| D | 8 à 10 | il n'en a jamais vu |

Avec 1 % de tokens à 9 au lieu de 0 : 0,01 × 9 ≈ 0,09 de perte en plus. Un si petit pourcentage creuse un écart visible.

## Résultats (800 fenêtres régulières, CPU, checkpoints à l'étape 20 000)

| | Validation non filtrée | Validation filtrée | Effet du bruit |
|---|---|---|---|
| A | 3,0556 | 3,0794 | −0,024 (avantagé) |
| D | 3,0931 | 3,0777 | +0,016 (pénalisé) |
| **D − A** | **+0,038** | **−0,002** | |

L'écart de 0,038 est expliqué à 100 % par le bruit de la validation (0,024 + 0,016). Les mesures utilisent 800 fenêtres : −0,002 est à lire comme « à égalité ».

Sortie complète : `logs/validation_filtree_sortie.txt`.

## Ce qu'on en déduit

- **Règle de méthode** : si l'on change le traitement des données d'entraînement, il faut aussi évaluer sur une validation traitée de la même façon. Sinon, la comparaison favorise le modèle qui a vu le bruit.
- Le filtrage n'a pas amélioré le français du modèle : le bruit (moins de 2 % des caractères) était trop faible pour peser. D reste utile comme exercice de méthode.
- On garde A comme base. La validation non filtrée reste la référence pour comparer à 0, B et A ; la version filtrée (`validation_v2.bin`) sert seulement aux comparaisons qui touchent au filtrage.
- Pour ne pas surcharger le CPU pendant un entraînement, le script mesure 800 fenêtres plutôt que toute la validation (environ 30 s par modèle et par validation, au lieu de plus d'une heure).
