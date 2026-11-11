# Exercices : le tokenizer de mots

Cours associé : [`../scripts/02_mots.py`](../scripts/02_mots.py) (module : [`../tools/tokenizer_basic.py`](../tools/tokenizer_basic.py))

Suite de la fiche 01 : un token n'est plus un caractère mais un **mot**. Petit corpus de 39 mots (8 phrases sur des animaux) :

```text
le chat dort. le chien dort aussi. le chat mange. le chien mange un os.
un oiseau chante. le chat regarde l'oiseau. le chien court. l'oiseau dort.
```

## 1. Découper en mots

Règles choisies : tout en minuscules, un mot = une suite de lettres, chaque signe de ponctuation est un token à part.

```text
['le', 'chat', 'dort', '.', 'le', 'chien', 'dort', 'aussi', '.', ...]
```

Ces règles sont arbitraires : un autre tokenizer garderait les majuscules, ou découperait « l'oiseau » autrement.

## 2. Le vocabulaire

16 tokens : les 15 mots distincts du corpus (`.` et `'` compris) plus un token spécial **`<inc>`** (identifiant 0), qui remplace tout mot absent du corpus.

## 3. Encoder, décoder, et le mot inconnu

| texte | identifiants | décodé |
|-------|--------------|--------|
| `le chien dort.` | `[10, 6, 8, 2]` | `le chien dort .` |
| `le chat ronronne.` | `[10, 5, 0, 2]` | `le chat <inc> .` |

`ronronne` n'est pas dans le corpus : il devient `<inc>`, et le mot est **perdu pour toujours** (impossible de le retrouver au décodage).

## 4. Mots contre caractères sur un texte jamais vu

Texte : `le canard nage. le chat dort. un poisson nage.`

| découpage | tokens | inconnus |
|-----------|--------|----------|
| mots | 12 | 4 (33 %) |
| caractères (leçon 1) | 46 | 1 (2 %) |

## 5. Comment le vocabulaire grandit

| mots lus | mots distincts | caractères lus | caractères distincts |
|----------|----------------|----------------|----------------------|
| 10 | 6 | 39 | 15 |
| 20 | 9 | 78 | 17 |
| 39 | 15 | 158 | 18 |

Les mots distincts augmentent avec le texte. Les caractères distincts se stabilisent vite.

## Ce qu'on en déduit

- **Les mots donnent des séquences courtes** : 12 tokens contre 46 pour la même phrase en caractères. Le réseau voit plus loin avec le même nombre de tokens.
- **Mais le mot inconnu est le défaut majeur** : `<inc>` efface l'information. Avec un corpus immense il devient rare, sans jamais disparaître (noms propres, fautes, mots techniques). Pour notre corpus « technique », ce serait un vrai problème.
- **Le vocabulaire de mots grossit sans fin** : plusieurs centaines de milliers de mots en français, avec accords et conjugaisons (« mange », « mangent », « mangeait »…). La couche d'entrée et de sortie du GPT deviendraient énormes pour la GTX 1070.
- **Les caractères se stabilisent** (~30 en français, un peu plus avec accents et majuscules), mais donnent de longues séquences.
- **Conclusion** : ni l'un ni l'autre n'est idéal. Les **sous-mots** (morceaux de mots : `chats` = `chat` + `s`) prennent le meilleur des deux : un vocabulaire de taille fixe, des séquences raisonnables, et aucun mot inconnu puisqu'on peut toujours retomber sur des morceaux ou des caractères. Ils sont trouvés par **BPE** (*Byte Pair Encoding*) : on part des caractères et on fusionne à répétition la paire de tokens voisins la plus fréquente. C'est la leçon suivante.
