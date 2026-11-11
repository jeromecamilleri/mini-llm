# 04 — Entraîner notre mini-GPT

## Objectif global

Entraîner localement un modèle de langage qui **prédit le token suivant**, avec une boucle d'entraînement écrite explicitement.

## À la fin de ce chapitre, on saura

- découper des textes en séquences, avec des cibles décalées d'un token ;
- calculer la perte (Cross Entropy) et mettre à jour les paramètres (AdamW) ;
- enregistrer des checkpoints et des métriques ;
- évaluer sur des données de validation (perte, perplexité) ;
- obtenir un modèle capable de générer du texte token après token.

## Parcours

| Leçon | Sujet | État |
|---|---|---|
| 1 | le GPT en PyTorch, perte de départ, gradient | fait |
| 2 | la boucle d'entraînement (AdamW, validation, génération) | fait |
| 3 | checkpoints, meilleur modèle, métriques | fait |

## Organisation

`scripts/` (scripts), `fiches/` (fiches `.md`), `tools/` (`gpt_torch.py` : le GPT PyTorch ; `donnees.py` : corpus et batchs ; `checkpoint.py` : sauvegarde et rechargement).

Lancement : `.venv/bin/python src/04_training/scripts/01_gpt_pytorch.py`

## Prérequis

Chapitres `01_fundamentals` (boucle d'entraînement, validation, arrêt anticipé), `02_tokenizer` et `03_transformer`. Pour l'instant le corpus est `02_tokenizer/corpus_animaux.txt` (67 mots) ; les vrais corpus viendront du dossier `datasets/`.

## Pour illustrer : la boucle d'entraînement

Les blocs 4 à 6 et 9 à 11 de cette page (pertes, rétropropagation, Adam, courbes d'entraînement et de validation, régularisation, hyperparamètres) résument ce chapitre (image externe, source MLTUT, non reproduite par nous).

![Deep Learning Cheat Sheet](../../img/Image%20ChatGPT%208%20nov.%202026,%2021_18_50.jpg)
