# 05 — Un tokenizer BPE sur les livres

Script : [`../scripts/05_bpe_corpus.py`](../scripts/05_bpe_corpus.py) · outil : [`../tools/bpe_rapide.py`](../tools/bpe_rapide.py)

## L'essentiel en 30 secondes

- Un **tokenizer** découpe le texte en morceaux de mots (tokens) ; le modèle ne voit que des numéros de tokens.
- Le BPE de `02_tokenizer` marchait sur 67 mots. Sur 36 millions de mots il serait beaucoup trop lent.
- Deux accélérations : compter **chaque mot différent une seule fois** (avec sa fréquence) et ne mettre à jour que les mots touchés par une fusion ; à l'encodage, **mémoriser** le découpage de chaque mot.
- Le résultat est écrit dans `datasets/tokenized/bpe_4000.json`.

## Pourquoi c'est plus rapide

| | BPE de `02_tokenizer` | BPE rapide |
|---|---|---|
| comptage des paires | tout recompter à chaque fusion | mis à jour seulement pour les mots concernés |
| choix de la meilleure paire | `most_common` sur tout | tas (liste triée par fréquence) |
| encodage | rejouer toutes les fusions sur chaque mot | fusion la plus ancienne d'abord, résultat mémorisé |

Idée clé : « ␣de » apparaît des millions de fois mais a toujours le même découpage. On le traite une fois et on multiplie par sa fréquence.

## Le réglage : le nombre de fusions

Vocabulaire final = 1 token `<inc>` + environ 110 caractères + fusions. Avec 4 000 fusions : environ 4 100 tokens.

| Vocabulaire | Séquences | Embeddings |
|---|---|---|
| petit | plus longues (plus de calcul) | petits |
| grand | plus courtes | plus gros, tokens rares mal appris |

## Vérifications faites

- Encoder puis décoder redonne **exactement** le texte (testé sur un extrait de Voltaire).
- Comparé au BPE lent sur 60 000 caractères et 300 fusions : 135 fusions identiques sur 300. Les écarts viennent des ex æquo (deux paires aussi fréquentes : l'ordre de choix diffère). Le nombre de tokens obtenu est quasi identique (26 196 contre 26 199).

## À observer quand tu lances le script

- Combien de mots **différents** et combien n'apparaissent qu'**une seule fois** (noms propres, fautes d'OCR).
- Les premières fusions (syllabes, lettres) et les dernières (mots entiers, noms propres).
- Le nombre de **caractères par token** : plus il est grand, plus le BPE compresse.
- Comment « Zorglubissime » (mot inventé) est découpé en morceaux connus.

## Ce qu'on en déduit

- Un mot inventé ou inconnu ne pose pas de problème : il se découpe en sous-mots, voire en caractères.
- Le tokenizer est entraîné sur les livres du corpus : sur un autre type de texte (technique, anglais) il découpera moins bien. Cela comptera pour les expériences A à D.
- Étape suivante : tokeniser tout le corpus avec ce BPE, en gardant des livres entiers à part pour la validation.
