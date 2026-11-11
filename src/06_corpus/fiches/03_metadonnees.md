# 03 — Métadonnées et statistiques par livre

Script : [`../scripts/03_metadonnees.py`](../scripts/03_metadonnees.py) · outil : `lire_metadonnees` dans [`../tools/mobi.py`](../tools/mobi.py)

## L'essentiel en 30 secondes

- **Métadonnées** = les données *sur* un livre (titre, auteur, langue, taille), pas son texte.
- Le fichier Kindle les contient déjà (bloc « EXTH ») : on les lit à la main, sans deviner d'après le nom du fichier.
- On mesure aussi le texte nettoyé et on écrit une ligne par livre dans `datasets/metadata.csv`.
- Ce fichier servira à choisir quels livres mettre dans quelle expérience.

## Colonnes de `metadata.csv`

| Colonne | Origine |
|---|---|
| `titre`, `auteur`, `editeur`, `langue_declaree` | lus dans le fichier Kindle |
| `format` | `mobi` ou `azw3` |
| `langue_estimee` | calculée : plus de « le, la, de… » que de « the, of, and… » ? |
| `caracteres`, `mots`, `paragraphes` | mesurés sur le texte nettoyé |
| `mots_differents` | vocabulaire du livre |
| `lettres_par_mot` | longueur moyenne d'un mot |

## Résultats sur les 3 livres

| Livre | Mots | Mots différents | Diversité |
|---|---|---|---|
| Bergounioux, *Faute d'égalité* | 6 927 | 2 732 | 39 % |
| De Luca, *Le poids du papillon* | 13 269 | 3 160 | 24 % |
| Voltaire, *L'Ingénu* | 24 002 | 4 559 | 19 % |

Total : 44 198 mots, 258 155 caractères. Langue déclarée et langue estimée concordent partout.

## Ce qu'on en déduit

- La **diversité** (mots différents / mots) baisse quand le livre s'allonge : plus on lit, plus on recroise les mêmes mots. Il ne faut donc comparer que des livres de longueur proche, ou regarder le corpus entier.
- Le comptage par expression régulière (6 927) diffère du simple `split()` de la leçon 2 (7 025) : « l'heure » compte pour un mot ici et la ponctuation est écartée. Un « mot » dépend de la définition choisie ; le vrai découpage viendra du tokenizer BPE.
- Les 3 livres font 44 000 mots : c'est très peu pour un GPT. Les 353 livres devraient en donner quelques dizaines de millions.
- `langue_estimee` permettra de repérer les livres en anglais ou mal étiquetés avant l'entraînement.
- `metadata.csv` est ignoré par git, comme les livres.
