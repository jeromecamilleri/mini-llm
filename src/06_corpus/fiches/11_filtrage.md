# 11 — Filtrer le bruit du corpus (expérience D)

Script : [`../scripts/11_filtrage.py`](../scripts/11_filtrage.py), outil : [`../tools/filtrage.py`](../tools/filtrage.py)

## L'essentiel en 30 secondes

- Un modèle apprend ce qu'on lui montre, **déchets compris**. Un corpus de romans contient aussi du code de feuille de style, des ISBN, des adresses web.
- On mesure d'abord le bruit, puis on écrit une copie filtrée dans `datasets/cleaned_v2/`. `datasets/cleaned/` n'est jamais modifié.
- Mesuré : 1,93 % des caractères retirés (221 M → 216 M). Le bruit est faible : n'attends pas un miracle de l'expérience D.

## Les règles (une ligne = un paragraphe)

| Règle | Ce qu'elle retire | Lignes retirées |
|---|---|---|
| code web | `@font-face`, `{ }`, balises `<...>` : des feuilles de style restées dans les livres | 27 176 |
| peu de lettres | moins de la moitié de lettres (hors répliques) : numéros, références | 8 318 |
| édition | ISBN, ©, adresses web, « Table of Contents » | 3 858 |
| code isolé | `FL4103`, `U000`, « Dynamic layout » | 2 205 |
| alphabet rare | plus de 10 % de caractères rares : grec, coréen, symboles | 314 |
| caractères rares | quelques caractères rares dans une ligne normale : on retire juste ces caractères | 3 223 |

Alphabet gardé : les caractères vus au moins 300 fois, soit 136 sur 611. Avec un seuil de 2 000, on perdait des lettres utiles dans les noms étrangers (`ü`, `í`, `ó`).

## Le piège de la comparaison

Si on filtrait aussi la validation, la perte porterait sur un autre texte : plus facile, donc non comparable à la référence et à B. On filtre **seulement l'entraînement** ; la validation reste celle des livres non filtrés (leçon 12).

## Ce qu'on en déduit

- **Pourquoi filtrer** : chaque token de code web est un token d'entraînement qui n'apprend rien sur le français. Le gain direct est petit (3 % de tokens en moins), mais le modèle cesse de produire de faux CSS ou de faux ISBN.
- **Pourquoi des règles simples** : visibles, vérifiables à l'œil dans la sortie, réversibles (la source reste intacte).
- **Alternatives existantes** : en pratique, les grands corpus utilisent des filtres de qualité (perplexité d'un petit modèle, dédoublonnage de documents, détection de langue). Notre baromètre de prévisibilité pourrait jouer ce rôle.
- **Lien avec le GPT final** : la qualité des données pèse autant que la taille du modèle. Ce sera à mesurer : D change une seule chose, le texte d'entraînement.
- Limite : les règles laissent passer le bruit que je n'ai pas vu (tables des matières sans numéro, mentions d'éditeur en phrase normale).
