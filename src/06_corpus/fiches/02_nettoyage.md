# 02 — Nettoyer le texte extrait

Script : [`../scripts/02_nettoyage.py`](../scripts/02_nettoyage.py) · outil : [`../tools/nettoyage.py`](../tools/nettoyage.py)

## L'essentiel en 30 secondes

- Un modèle apprend **tout** ce qu'on lui montre, déchets compris.
- On applique 4 règles simples, une par une : retirer l'en-tête Gutenberg, normaliser les symboles, réparer les lettrines, refaire les paragraphes.
- Le résultat est écrit dans `datasets/cleaned/`.

## Les règles

| Règle | Avant | Après |
|---|---|---|
| `retirer_gutenberg` | licence anglaise de 15 600 caractères | supprimée |
| `normaliser` | `’` `“` `”` `—` `…` espace insécable | `'` `"` `-` `...` espace |
| `reparer_lettrines` | `Àl'heure` | `À l'heure` |
| `refaire_paragraphes` | lignes coupées tous les 70 caractères | un paragraphe = une ligne |

Pourquoi normaliser ? `’` et `'` disent la même chose mais seraient deux symboles différents : le modèle devrait apprendre deux fois la même chose.

## Résultats sur les 3 livres

| Livre | Mots avant | Mots après | Paragraphes |
|---|---|---|---|
| Bergounioux | 7 024 | 7 025 | 157 → 79 |
| De Luca | 13 255 | 13 254 | 505 → 252 |
| Voltaire | 25 963 | 23 443 | 3 161 → 344 |

Le corpus nettoyé compte 106 caractères différents.

## Ce qu'il reste

- Voltaire commence encore par le bloc « Produced by… » et la page de titre de l'imprimeur.
- De Luca commence par la page de titre (« ERRI DE LUCA / LE POIDS… »).
- Quelques symboles rares : `©`, `°`, `@`, des majuscules accentuées isolées.
- On ne corrige pas les erreurs d'OCR de Voltaire (« N° 2O »).

## Ce qu'on en déduit

- Le nettoyage coûte peu de lignes mais change beaucoup de choses : 90 % des « paragraphes » de Voltaire n'étaient que des retours à la ligne.
- Chaque livre a ses propres défauts : sur 353 livres, il faudra regarder des échantillons et ajouter des règles au fur et à mesure.
- Règle d'or : on écrit toujours un **nouveau** fichier, les originaux ne sont jamais modifiés.
