# Exercices : du texte aux nombres (caractère par caractère)

Cours associé : [`../scripts/01_caracteres.py`](../scripts/01_caracteres.py) (module : [`../tools/vocabulary.py`](../tools/vocabulary.py))

On quitte le XOR : ici le fil rouge est un **mini-texte** écrit à la main.

```text
le chat dort. le chien dort aussi. le chat mange.
```

## Le problème

Notre réseau ne sait traiter que des nombres (les `x1`, `x2` du XOR). Un GPT reçoit du **texte**. Il faut donc une traduction dans les deux sens :

```text
texte → identifiants (nombres entiers) → réseau → identifiants → texte
```

Cette traduction s'appelle la **tokenisation**. Un **token** est une unité de texte. Le **vocabulaire** est la liste des tokens connus, chacun avec son identifiant.

## Avec des caractères

Un token = un caractère. Le texte fait 49 caractères, dont **17 distincts** : c'est le vocabulaire.

| token | `␣` | `.` | `a` | `c` | `d` | `e` | `g` | `h` | `i` |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |

| token | `l` | `m` | `n` | `o` | `r` | `s` | `t` | `u` |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| id | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |

(`␣` est l'espace, un token comme les autres.)

## Encoder et décoder

| texte | identifiants |
|-------|--------------|
| `le chat` | `[9, 5, 0, 3, 7, 2, 15]` |

Décoder ces 7 identifiants redonne `le chat` : l'aller-retour est exact, on ne perd rien. Sur tout le corpus : 49 identifiants.

## Un caractère inconnu

Le vocabulaire est figé : il ne contient que ce qu'il a vu. Encoder `le chat ronronne é` échoue, car **`é` n'est pas dans le vocabulaire**.

## Un identifiant n'a pas de sens

Les identifiants sont attribués par ordre alphabétique : `a` = 2 et `c` = 3 sont voisins, `a` et `u` = 16 sont éloignés, sans aucun rapport avec leur sens. Donner « 3 » au réseau lui ferait croire que `c` est « entre » `a` et `e`.

Une solution simple est le **one-hot** : pour chaque caractère, un vecteur de 17 cases avec un seul 1.

| lettre `l` (id 9) | `[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]` |
|---|---|

Le texte `le chat` devient un tableau de forme (7, 17) : 7 caractères, 17 cases chacun. Mais tous les caractères sont à égale « distance » : le réseau ne sait pas que `a` et `e` sont des voyelles. Ce sont les **embeddings** (plus tard dans ce labo) qui apprendront un petit vecteur par token, avec du sens.

## Caractères contre mots

| découpage | tokens pour le texte | taille du vocabulaire |
|-----------|----------------------|-----------------------|
| caractères | 49 | 17 |
| mots | 13 | 7 |

## Ce qu'on en déduit

- **Un modèle de langage ne lit pas du texte, il lit des identifiants.** Tout ce qu'on fera ensuite (le GPT, l'entraînement) travaille sur des listes d'entiers.
- **Caractères : petit vocabulaire, longues séquences.** Peu de tokens différents (17), mais 49 positions pour 13 mots : le réseau doit relier beaucoup de tokens pour « voir » un mot.
- **Mots : séquences courtes, vocabulaire qui explose.** 7 mots ici, mais plusieurs centaines de milliers dans une vraie langue, et un mot jamais vu est impossible à encoder.
- **Les sous-mots sont le compromis** utilisé par les vrais GPT. Un **sous-mot** est un morceau de mot, entre le caractère et le mot entier : `chats` devient `chat` + `s`, et un mot rare est découpé en morceaux connus. Le vocabulaire est de taille fixe, et aucun mot n'est inconnu. Ces morceaux sont trouvés automatiquement par la méthode **BPE** (*Byte Pair Encoding*) : on part des caractères et on fusionne les paires les plus fréquentes. On les construit à la leçon 3 : pas besoin de les comprendre maintenant.
- **Le vocabulaire pèse sur le modèle** : plus il est grand, plus la couche d'entrée (embeddings) et de sortie sont grosses. Avec 17 caractères elles sont minuscules, ce qui convient à notre GTX 1070.
- **Le vocabulaire se construit sur le corpus d'entraînement** : changer de corpus (français, technique, mixte) change le vocabulaire, donc les identifiants.
