# 05 — Inférence et expérimentation

## Objectif global

Observer le comportement du modèle entraîné et comprendre les facteurs qui l'influencent, pour relier ce qu'il génère aux données et aux paramètres utilisés.

## À la fin de ce chapitre, on saura

- générer du texte token après token (génération autorégressive) ;
- jouer sur la température, le Top-K et le Top-P ;
- comparer plusieurs modèles sur des prompts identiques ;
- relier la qualité du texte aux métriques d'entraînement et de validation.

## Parcours

| Leçon | Sujet | État |
|---|---|---|
| 1 | génération autorégressive avec un checkpoint | fait |
| 2 | température, Top-K, Top-P | fait |
| 3 | comparer des modèles sur les mêmes prompts | fait |

## Organisation

`scripts/` (scripts), `fiches/` (fiches `.md`), `tools/` (`generation.py` : chargement du modèle, probabilités, filtres, boucle de génération).

Lancement : `.venv/bin/python src/05_inference/scripts/01_generation.py`

## Prérequis

Chapitre `04_training` (un modèle entraîné et ses checkpoints : lancer d'abord `scripts/03_checkpoints.py`).

## Pour illustrer : prédire puis générer

Les blocs 8 et 9 (prédiction du token suivant, génération pas à pas) décrivent ce que fait ce chapitre (image externe, auteure Pooja Pawar, non reproduite par nous).

![Large Language Model, vue d'ensemble](../../img/Image%20ChatGPT%208%20nov.%202026,%2021_18_15.jpg)
