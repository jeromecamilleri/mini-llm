# Leçon 7 : le petit GPT complet

Script : [`07_petit_gpt.py`](../scripts/07_petit_gpt.py)
Code : [`tools/gpt.py`](../tools/gpt.py)
Lancement : `.venv/bin/python src/03_transformer/scripts/07_petit_gpt.py`

## Le sigle GPT

| Lettre | Mot | Sens | Où dans le projet |
|---|---|---|---|
| **G** | Generative | il **génère** du texte, un mot après l'autre | `05_inference` |
| **P** | Pre-trained | **pré-entraîné** sur un énorme corpus à prédire le mot suivant, avant d'être adapté à une tâche | `04_training` |
| **T** | Transformer | l'**architecture** : attention + feed forward, empilés | `03_transformer` |

Le « Chat » de ChatGPT désigne l'interface de conversation, pas l'architecture.

## L'essentiel en 30 secondes

Un GPT, c'est le tuyau :

```
mots -> identifiants -> vecteurs + positions -> bloc 1 -> ... -> bloc N -> un score par mot
```

Tout a déjà été vu : tokenizer (chapitre 02), embeddings et positions (leçon 4), blocs (leçon 6). Il ne reste que la **dernière couche** : elle transforme les `d` nombres de chaque position en **un score par mot du vocabulaire**.

Ici les poids sont **au hasard** : le modèle n'a rien appris, donc il génère du charabia. C'est le rôle du chapitre `04_training`.

## Les formes

| Étape | Forme | Sens |
|---|---|---|
| identifiants | `(T,)` | T mots |
| embeddings + positions | `(T, d)` | chaque mot devient d nombres |
| après chaque bloc | `(T, d)` | même forme : on peut empiler |
| scores par mot | `(T, V)` | V = taille du vocabulaire |

La ligne `t` répond à : « quel mot vient **après** la position `t` ? » Pour générer, on ne garde que la **dernière** ligne.

## Générer, mot par mot

1. On donne un début : `le chat`.
2. Le GPT calcule les scores ; on regarde la dernière ligne.
3. On choisit un mot : le plus probable (**glouton**) ou un tirage au hasard selon les probabilités.
4. On l'ajoute à la phrase et on recommence.

### Glouton ou tirage ?

Après `le chat`, le modèle donne par exemple : `dort` 50 %, `mange` 30 %, `court` 20 %.

| Méthode | Règle | Résultat |
|---|---|---|
| **Glouton** (*greedy*) | on prend toujours le mot le **plus probable** | toujours `dort` |
| **Tirage** (*sampling*) | on tire au sort selon les probabilités | `dort` 5 fois sur 10, `mange` 3 sur 10, `court` 2 sur 10 |

- Glouton : toujours le même résultat, sûr, mais répétitif ; il peut boucler (`aussi mange aussi aussi…`).
- Tirage : varié et créatif, mais peut choisir un mot peu probable et dérailler. C'est pourquoi ChatGPT ne répond pas deux fois pareil.

### La température

C'est un **réglage du tirage** : on divise les scores par `T` avant le softmax (`softmax(scores / T)`).

| Température | Effet | Exemple (scores 2,0 / 1,5 / 1,0) |
|---|---|---|
| basse (0,5) | écarte les probabilités : le favori domine, on tend vers le glouton | 67 % / 24 % / 9 % |
| 1 | on suit les probabilités du modèle | 51 % / 31 % / 19 % |
| haute (2) | aplatit : tous les mots deviennent presque aussi probables | 42 % / 33 % / 25 % |

Image : la température d'une casserole. Froid : les mots restent « figés » sur le favori. Chaud : ils s'agitent et tout devient possible.

Les **scores** ne changent pas, seule la façon de les transformer en probabilités change. Dans `generer`, `rng=None` donne le glouton ; sinon c'est le tirage, réglé par `temperature`.

### Peut-on la choisir dans les chatbots ?

| Interface | Température réglable ? |
|---|---|
| API OpenAI (développeurs), Playground, API d'autres fournisseurs (Anthropic, Mistral…) | **oui**, paramètre `temperature` (souvent de 0 à 2) |
| modèles locaux (llama.cpp, Ollama, LM Studio) | **oui**, réglage standard |
| ChatGPT (application grand public) | **non**, fixée par OpenAI |

Certains modèles récents de raisonnement imposent la température, même par l'API (non vérifié pour les modèles actuels).

| Usage | Température habituelle |
|---|---|
| code, maths, extraction de données | basse (0 à 0,3) |
| conversation | 0,7 à 1 |
| écriture créative, idées | 1 et plus |

Un réglage voisin, `top_p` : on ne tire que parmi les mots les plus probables, jusqu'à cumuler p % de probabilité. Ils seront vus dans `05_inference`, avec la température réglable du mini-GPT.

## Ce qu'on a lancé

Vocabulaire V = 13, d = 8, 2 têtes, 2 blocs, 1888 paramètres.

| Mesure | Résultat |
|---|---|
| formes | `(4, 8)` après chaque bloc, `(4, 13)` en sortie |
| perte d'un modèle au hasard | 2,381 |
| ln(13) | 2,565 |
| glouton | `le chat dort . aussi mange aussi aussi chat .` |
| tirage | `le chat dort oiseau ' oiseau aussi cheval le aussi` |
| changer le dernier mot | les scores du passé ne bougent pas (masque causal) |

## Ce qu'on en déduit

- Un GPT est une **fonction** : des identifiants entrent, un score par mot ressort, à chaque position.
- Un modèle qui ne sait rien a une perte proche de **ln(V)** (il hésite entre tous les mots). C'est le **point de départ** à battre à l'entraînement.
- Avec des poids au hasard, le texte est du charabia : la **structure** ne suffit pas, il faut **apprendre les poids**.
- Le masque causal tient dans le modèle complet : le futur n'influence jamais le passé.
- Le nombre de paramètres vient surtout de d (en d²) et du nombre de blocs.

## Et ensuite ?

Le chapitre `04_training` : mesurer la perte, calculer les gradients, ajuster les poids (AdamW) pour que la perte descende bien sous ln(V).
