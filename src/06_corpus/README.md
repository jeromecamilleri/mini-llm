# 06 — Préparer le corpus

## Objectif global

Transformer les livres Kindle (`datasets/originals/`) en texte propre, prêt à être découpé en tokens.

## Parcours

| Leçon | Sujet | État |
|---|---|---|
| 1 | extraire le texte d'un `.mobi` / `.azw3` | fait (3 livres) |
| 2 | nettoyer (en-têtes, lettrines, espaces) | fait (3 livres) |
| 3 | métadonnées et statistiques par livre | fait |
| 4 | repérer les livres suspects (triage) | fait |
| 5 | tokenizer BPE rapide sur le corpus | fait |
| 6 | tokeniser le corpus, livres de validation | fait |
| 7 | entraîner un GPT sur les livres | fait (validation 3,321, perplexité 28) |
| 8 | générer du texte (température, Top-K, Top-P) | fait |
| 9 | tracer la courbe d'apprentissage | fait |
| 10 | parler au modèle (chat simplifié) | fait |
| 11 | filtrer le bruit du corpus (expérience D) | fait |
| 12 | tokeniser le texte filtré (`train_v2.bin`) | fait |
| 13 | ce que le modèle a mis dans ses embeddings (cosinus, voisins, ACP) | fait |
| 14 | A et D sur une validation filtrée (le test était-il équitable ?) | fait |
| 15 | Perte sur toute la validation, fenêtre glissante, bits par octet | fait |
| 16 | Nettoyer le théâtre importé (Claire, TheatreGratuit) : format, didascalies, texte déformé | fait |
| 17 | Tokeniser le théâtre, réserver des dialogues de validation | fait |

Pas à pas complet du corpus : [`docs/howto_corpus_353_livres.md`](../../docs/howto_corpus_353_livres.md).

## Organisation

`scripts/`, `fiches/` (fiches `.md`), `tools/mobi.py` (lecture du format), `tools/nettoyage.py` (règles de nettoyage).

Lancement : `.venv/bin/python src/06_corpus/scripts/01_extraction.py`

Les livres sont protégés par le droit d'auteur : ils restent en local, jamais commités (voir `.gitignore`).
