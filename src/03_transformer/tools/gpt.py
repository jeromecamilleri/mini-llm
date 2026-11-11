"""Petit GPT complet (NumPy) : mots -> vecteurs -> blocs empilés -> score par mot.

Détail pas à pas dans scripts/07_petit_gpt.py. Réutilise attention.py et transformer_block.py.
"""
import numpy as np

from attention import softmax
from transformer_block import bloc_transformer, init_bloc, layer_norm


def init_gpt(taille_vocab, d, n_tetes, n_blocs, T_max, rng):
    """Tous les paramètres du GPT, tirés au hasard (petits nombres)."""
    return {
        "E": rng.normal(0, 0.3, (taille_vocab, d)),   # un vecteur par mot (leçon 4 du tokenizer)
        "P": rng.normal(0, 0.3, (T_max, d)),          # un vecteur par position (leçon 4)
        "blocs": [init_bloc(d, n_tetes, rng) for _ in range(n_blocs)],  # poids DIFFÉRENTS par bloc
        "W_sortie": rng.normal(0, 0.3, (d, taille_vocab)),  # vecteur final -> un score par mot
    }


def gpt(ids, p, trace=False):
    """ids : liste de T identifiants. Retourne les scores (logits), forme (T, taille_vocab).

        ids -> E[ids] + P[:T] -> bloc 1 -> ... -> bloc N -> layer_norm -> @ W_sortie

    La ligne t donne un score pour CHAQUE mot du vocabulaire : « quel mot après la position t ? »
    Avec trace=True, retourne aussi la forme du tenseur à chaque étape.
    """
    T = len(ids)
    X = p["E"][ids] + p["P"][:T]
    formes = [("embeddings + positions", X.shape)]
    for i, bloc in enumerate(p["blocs"], start=1):
        X = bloc_transformer(X, bloc)
        formes.append((f"bloc {i}", X.shape))
    logits = layer_norm(X) @ p["W_sortie"]
    formes.append(("scores par mot", logits.shape))
    return (logits, formes) if trace else logits


def nb_parametres(p):
    total = p["E"].size + p["P"].size + p["W_sortie"].size
    for b in p["blocs"]:
        total += sum(Wq.size + Wk.size + Wv.size for Wq, Wk, Wv in b["tetes"])
        total += b["Wo"].size + b["W1"].size + b["b1"].size + b["W2"].size + b["b2"].size
    return total


def perte(ids, p):
    """Entropie croisée moyenne : -log(probabilité donnée au VRAI mot suivant)."""
    logits = gpt(ids[:-1], p)
    probas = softmax(logits)
    vrais = np.array(ids[1:])
    return float(-np.log(probas[np.arange(len(vrais)), vrais]).mean())


def generer(ids, p, n_nouveaux, T_max, rng=None, temperature=1.0):
    """Ajoute n_nouveaux mots, un par un.

    À chaque pas : on lance le GPT sur les derniers mots (au plus T_max), on prend les scores
    de la DERNIÈRE position, puis on choisit un mot. rng=None -> le plus probable (glouton) ;
    sinon tirage au hasard selon les probabilités.
    """
    ids = list(ids)
    for _ in range(n_nouveaux):
        logits = gpt(ids[-T_max:], p)[-1]
        if rng is None:
            suivant = int(np.argmax(logits))
        else:
            suivant = int(rng.choice(len(logits), p=softmax(logits / temperature)))
        ids.append(suivant)
    return ids
