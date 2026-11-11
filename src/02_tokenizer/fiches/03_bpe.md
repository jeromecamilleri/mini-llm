# Exercices : BPE, les sous-mots trouvés automatiquement

Cours associé : [`../scripts/03_bpe.py`](../scripts/03_bpe.py) (module : [`../tools/bpe.py`](../tools/bpe.py))

Même corpus que la fiche 02. On a vu les caractères (petit vocabulaire, longues séquences) et les mots (séquences courtes, mots inconnus). Un **sous-mot** est un morceau de mot, entre les deux ; **BPE** (*Byte Pair Encoding*) les trouve automatiquement.

## L'algorithme

On part des caractères, puis on répète :

1. compter chaque **paire** de tokens voisins dans le corpus ;
2. **fusionner** la paire la plus fréquente en un nouveau token ;
3. ajouter ce token au vocabulaire.

Le nombre de fusions choisi fixe la taille finale du vocabulaire : **caractères + fusions**.

Le texte est d'abord coupé en morceaux : un mot garde l'espace qui le précède (`␣chat`), la ponctuation est à part. Les fusions ne traversent jamais un morceau.

## Les 15 premières fusions

Au départ, la paire la plus fréquente est `␣` + `c` (8 fois : `␣chat`, `␣chien`, `␣chante`, `␣court`…).

| n° | fusion | nouveau token | vue |
|----|--------|---------------|-----|
| 1 | `␣` + `c` | `␣c` | 8 fois |
| 2 | `␣c` + `h` | `␣ch` | 7 |
| 3 | `␣` + `l` | `␣l` | 7 |
| 4 | `␣l` + `e` | `␣le` | 5 |
| 5 | `␣ch` + `a` | `␣cha` | 4 |
| 6 | `r` + `t` | `rt` | 4 |
| 7 | `a` + `u` | `au` | 4 |
| 8 | `␣cha` + `t` | `␣chat` | 3 |
| 9 | `␣` + `d` | `␣d` | 3 |
| 10 | `␣d` + `o` | `␣do` | 3 |
| 11 | `␣do` + `rt` | `␣dort` | 3 |
| 12 | `␣ch` + `i` | `␣chi` | 3 |
| 13 | `␣chi` + `e` | `␣chie` | 3 |
| 14 | `␣chie` + `n` | `␣chien` | 3 |
| 15 | `o` + `i` | `oi` | 3 |

Les mots fréquents (`␣chat`, `␣dort`, `␣chien`) se construisent petit à petit, chaque fusion s'appuyant sur les précédentes. Vocabulaire final : 18 caractères + 15 fusions = **33 tokens**.

## Encoder avec BPE

Pour un texte neuf, on **rejoue les fusions dans l'ordre** où elles ont été apprises.

| texte | sous-mots | tokens |
|-------|-----------|--------|
| `le chat dort.` | `l` `e` `␣chat` `␣dort` `.` | 5 |
| `le chien ronronne.` | `l` `e` `␣chien` `␣` `r` `o` `n` `r` `o` `n` `n` `e` `.` | 13 |

- `le chat dort.` (mots connus) tient en 5 tokens au lieu de 13 caractères.
- `ronronne` n'existe pas dans le corpus mais n'est **pas inconnu** : il retombe sur des caractères, qui sont tous dans le vocabulaire. Rien n'est perdu, et le décodage redonne exactement `le chien ronronne.`. À comparer avec `<inc>` de la fiche 02.

## Combien de fusions ?

Tokens pour le corpus (146 caractères) et pour un texte neuf (`le canard dort. le chien nage.`, 30 caractères) :

| fusions | vocabulaire | corpus | texte neuf |
|---------|-------------|--------|------------|
| 0 (= fiche 01) | 18 | 146 | 30 |
| 5 | 23 | 115 | 25 |
| 15 | 33 | 83 | 18 |
| 30 | 48 | 55 | 17 |
| 60 | 64 | 39 | 17 |

## Ce qu'on en déduit

- **BPE n'est pas écrit à la main** : le corpus décide des sous-mots. Un corpus français et un corpus technique donneront des vocabulaires différents, donc des identifiants différents : à garder en tête pour les expériences A à D.
- **Aucun mot inconnu** : au pire, un mot est découpé jusqu'aux caractères. C'est le défaut des mots (`<inc>`) qui disparaît.
- **Un réglage à choisir : le nombre de fusions.** Plus il y en a, plus les séquences raccourcissent (146 → 39 tokens), mais plus le vocabulaire grossit (couche d'entrée et de sortie plus grosses). Il faudra le choisir selon notre corpus et la GTX 1070.
- **Un gain moins fort sur un texte neuf** (30 → 17 contre 146 → 39) : seuls les mots du corpus sont bien compressés. D'où l'intérêt d'un corpus d'entraînement assez grand.
- **Les vrais GPT** utilisent exactement ce principe, avec des dizaines de milliers de fusions (50 257 tokens pour GPT-2), et partent des **octets** (256 valeurs possibles) plutôt que des caractères, ce qui garantit qu'aucun texte ne pose problème.
- **Reste dans ce chapitre** : les embeddings (un petit vecteur appris par token) et les séquences d'entraînement.
