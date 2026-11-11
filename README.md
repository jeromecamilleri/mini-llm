# mini-llm

Laboratoire pédagogique : construire un petit modèle de langage de type GPT **à la main** (NumPy, puis PyTorch), l'entraîner sur ses propres livres, et mesurer comment le corpus et les réglages changent le résultat.

Le plan complet est dans [`docs/Readme.md`](docs/Readme.md). Le suivi chiffré des expériences est dans [`docs/suivi_experiences_corpus.md`](docs/suivi_experiences_corpus.md).

## Parcours

| Dossier | Contenu |
|---|---|
| [`src/01_fundamentals`](src/01_fundamentals/README.md) | neurone, réseau, rétropropagation, optimiseurs |
| [`src/02_tokenizer`](src/02_tokenizer/README.md) | caractères, BPE, embeddings |
| [`src/03_transformer`](src/03_transformer/README.md) | attention jusqu'à un petit GPT (NumPy) |
| [`src/04_training`](src/04_training/README.md) | GPT PyTorch, boucle d'entraînement, checkpoints |
| [`src/05_inference`](src/05_inference/README.md) | génération, température, Top-K, Top-P |
| [`src/06_corpus`](src/06_corpus/README.md) | corpus de 346 livres, entraînement, expériences |

Chaque dossier contient `scripts/` (les leçons, à lancer), `fiches/` (une fiche explicative par leçon) et `tools/` (modules réutilisables).

## Lancer une leçon

```bash
pip install -r requirements.txt
python src/01_fundamentals/scripts/a_la_main/01_neurone.py
```

Matériel visé : GPU GTX 1070 (8 Go). Détails d'installation : [`docs/installation_pytorch.md`](docs/installation_pytorch.md).

## Données non publiées

Les livres, le texte nettoyé, les fichiers tokenisés et les modèles entraînés ne sont **pas** dans ce dépôt (droits d'auteur, taille). Seuls le code, les fiches, les logs et les illustrations (`img/`) sont suivis. Les chapitres 01 à 05 fonctionnent sans ces données ; le chapitre 06 demande vos propres livres dans `datasets/originals/`.
