# 16 — Nettoyer le théâtre importé

Script : [`../scripts/16_nettoyage_theatre.py`](../scripts/16_nettoyage_theatre.py)

## L'essentiel en 30 secondes

- Source : corpus **Claire** (Hugging Face), partie `TheatreGratuit`, fichier `datasets/claire/train.txt` : 155 066 répliques, 4 443 personnages, 6,76 M de tokens.
- Format d'origine : `[PERSONNAGE:] texte`. Notre modèle (et le chat) lit `- texte`.
- Le script convertit le format, retire les **didascalies** collées dans les répliques et le **texte déformé** (accents imités, bégaiements).
- Résultat : **150 799 répliques gardées (97,2 %)**, **5,02 M de tokens**, soit **7,0 %** du total si on les ajoute aux 66,2 M des livres.
- Licence **CC BY-NC-SA** : le dossier `datasets/claire/` est dans `.gitignore`, jamais commité.

## Ce que le script fait

| Règle | Exemple | Supprimées |
|---|---|---|
| Format : retirer le nom, écrire `- ` | `[VILDAMOUR:] Ah !` → `- Ah !` | (toutes les répliques sont converties) |
| Didascalie : un nom de personnage en MAJUSCULES dans la réplique | « Ecoute !... MARCELLE. — Non, non, ça suffit ! » | 3 974 (2,56 %) |
| Texte déformé : au moins 30 % de mots absents des livres (réplique de 5 mots ou plus) | « Oh ! eè éa-é-ae. » | 291 (0,19 %) |
| Réplique d'un caractère | `?` | 2 |

Les lignes vides entre scènes sont gardées (une seule à la fois).

## Pourquoi ces règles

- **Retirer les noms** : 4 443 noms distincts. Le modèle apprendrait des noms de pièces, pas à dialoguer.
- **Didascalies** : une indication de scène (« MARCELLE sort ») n'est pas une parole. Les cris (« NON ! ») restent, car seuls les noms de personnages du corpus déclenchent la règle.
- **Mots absents des livres** : le vocabulaire vient des 346 livres (mots vus au moins 5 fois : 100 208 mots), sans dictionnaire, comme pour la mesure des mots inventés. Aucun critère « lettres courtes » : il prenait « Qu'est-ce qu'il a ? » pour du texte cassé.

## Ce qui reste imparfait

- Les répliques de moins de 5 mots ne sont pas jugées : « E en ! e ein o-ésses e ou ites a ! » a survécu.
- 408 répliques de plus de 1 500 caractères sont gardées (monologues, mais aussi des scènes mal découpées).
- Quelques phrases en anglais, l'orthographe ancienne et quelques noms de scène non en majuscules.
- C'est du théâtre : situations et langue d'époque, pas de la conversation de tous les jours.

## Les chiffres

| | Tokens |
|---|---|
| Avant nettoyage (BPE 4 612 tokens, on ne le refait pas) | 6 761 976 |
| Après nettoyage | 5 020 817 (74 %) |
| Tokens inconnus après | 4 125 (0,08 %) |
| Livres d'entraînement | 66 216 164 |
| Part du théâtre dans le mélange | 7,0 % |

## Ce qu'on en déduit

- Le nettoyage retire **un quart des tokens**, surtout les noms de personnages entre crochets, pas les répliques : 97 % des répliques restent.
- À 7 %, un simple mélange aura probablement **un petit effet** sur la perte. Elle se mesure sur des livres, pas sur du dialogue. Il faudra une validation de dialogues, sinon le gain est invisible.
- Le BPE n'a pas été refait : si le théâtre apporte un gain, il vient bien du corpus et pas d'un autre tokenizer.
