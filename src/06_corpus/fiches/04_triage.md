# 04 — Repérer les livres suspects

Script : [`../scripts/04_triage.py`](../scripts/04_triage.py)

## L'essentiel en 30 secondes

- Avec 353 livres, on ne peut pas tout lire : on cherche les **anomalies** avec les chiffres de `metadata.csv`.
- Le script **propose** une liste de suspects, avec la raison et le début du texte. Il ne supprime rien.
- C'est toi qui décides, en écrivant les noms de fichiers à écarter dans `datasets/exclusions.txt`.

## Les critères

| Critère | Ce qu'il détecte |
|---|---|
| moins de 5 000 mots | extraction ratée, extrait gratuit |
| langue estimée ≠ français | livre en anglais |
| langue déclarée ≠ français | étiquette Kindle fausse (souvent sans gravité) |
| même titre et même auteur | doublon |
| diversité du vocabulaire hors de 3 % – 50 % | texte répétitif ou illisible |

La **diversité** = mots différents ÷ mots. Elle baisse quand le livre est long, d'où des seuils larges.

## Résultat : 27 suspects sur 353

| Cas | Livres | Décision |
|---|---|---|
| extraction ratée | *René Girard, Des choses cachées* (8 mots) | écarté |
| en anglais | *Animal Farm*, *Hernani*, *Phoenix Project* | écartés |
| doublons | Becky Chambers, *Central Station*, *Les Voyageurs tome 3* | une copie écartée chaque fois |
| étiquette fausse | une vingtaine de livres français | gardés |
| texte bruité | *Anéantir*, rapport « Lumières à l'ère numérique » | gardés |

`exclusions.txt` contient 7 fichiers : il reste **346 livres**.

## Ce qu'on en déduit

- Une mesure simple (nombre de mots, fréquence de « le / the ») suffit à trouver les vrais problèmes.
- Les étiquettes des fichiers ne sont pas fiables : on se fie au contenu.
- Un doublon fausserait l'entraînement : le modèle verrait deux fois le même livre et le **mémoriserait** plus facilement.
- La décision reste humaine : un script peut se tromper (fausses alertes sur la langue).
