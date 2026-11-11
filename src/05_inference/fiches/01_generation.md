# Leçon 1 : générer du texte avec le modèle entraîné

Script : [`01_generation.py`](../scripts/01_generation.py)
Code : [`tools/generation.py`](../tools/generation.py)
Lancement : `.venv/bin/python src/05_inference/scripts/01_generation.py`

## L'essentiel en 30 secondes

**Inférence** = utiliser un modèle **déjà entraîné**, sans le modifier (pas de gradient). On recharge le checkpoint de `04_training` (`models/checkpoints/animaux_meilleur.pt`) et on génère du texte, **un mot à la fois**.

La boucle **autorégressive** (« qui se nourrit de sa propre sortie ») :

| Étape | Ce qui se passe |
|---|---|
| 1 | on encode le début (le **prompt**) en identifiants |
| 2 | le modèle donne un score par mot ; on regarde la **dernière** position |
| 3 | softmax : scores → probabilités |
| 4 | on **choisit** un mot (glouton ou tirage au sort) |
| 5 | on l'ajoute au texte et on recommence à l'étape 2 |

Le modèle ne prédit jamais « la phrase » : seulement **un mot**, à chaque tour. Il n'a **pas de mémoire** entre deux tours : il relit le texte, dans la limite de sa **fenêtre** (`T_max = 4` mots ici).

## Ce qu'on a lancé

Prompt `le chat`, probabilités du mot suivant :

| Mot | Probabilité |
|---|---|
| dort | 41,3 % |
| court | 30,8 % |
| mange | 25,0 % |
| cheval, chien… | < 1 % chacun |

Boucle au ralenti, glouton :

| Le modèle voit | Choisit | Probabilité |
|---|---|---|
| `le chat` | dort | 41 % |
| `le chat dort` | . | 71 % |
| `le chat dort .` | le | 95 % |
| `chat dort . le` | cheval | 38 % |

Le texte final : `le chat dort . le cheval dort . le chat dort .`

| Méthode | Résultat depuis `le chat` |
|---|---|
| glouton | toujours la même phrase |
| tirage | `le chat mange . le chat chien mange . le cheval court` (faute !), puis d'autres phrases correctes |
| mot inconnu (`tigre`) | devient `<inc>` ; le modèle continue sans le comprendre |

## Ce qu'on en déduit

- Générer = **répéter** « prédire un mot, l'ajouter, recommencer ». Rien d'autre.
- Le glouton est toujours le même ; le tirage **varie**, mais peut produire une faute, car un mot peu probable a été tiré.
- Le modèle **oublie** ce qui sort de sa fenêtre : à `chat dort . le`, il choisit `cheval` sans savoir qu'on parlait d'un chat. Ici ce n'est pas grave (corpus très régulier) ; sur un vrai texte, une fenêtre longue est précieuse.
- Hors du vocabulaire (13 mots), le modèle ne comprend rien : `tigre` devient `<inc>`.
- Le checkpoint suffit : plus besoin de ré-entraîner pour générer.

## Et ensuite ?

Leçon 2 : régler le tirage (température, Top-K, Top-P).
