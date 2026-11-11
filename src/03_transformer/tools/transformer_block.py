"""Bloc Transformer : normalisation, feed forward, connexions résiduelles."""
import numpy as np

from attention import attention_multi_tetes


def layer_norm(X, gamma=1.0, beta=0.0, eps=1e-5):
    """Normalise CHAQUE token : moyenne 0 et écart-type 1 sur ses d nombres.

    Puis gamma (échelle) et beta (décalage), appris. Un token est traité seul.
    """
    moyenne = X.mean(axis=-1, keepdims=True)
    ecart = X.std(axis=-1, keepdims=True)
    return gamma * (X - moyenne) / (ecart + eps) + beta


def gelu(x):
    """ReLU « lissée » : celle des Transformers (GPT-2)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def feed_forward(X, W1, b1, W2, b2):
    """Petit réseau classique appliqué à CHAQUE token séparément : d -> 4d -> d."""
    return gelu(X @ W1 + b1) @ W2 + b2


def init_bloc(d, n_tetes, rng):
    """Paramètres d'un bloc, tirés au hasard (petits nombres)."""
    d_tete = d // n_tetes
    s = 0.3
    tetes = [tuple(rng.normal(0, s, (d, d_tete)) for _ in range(3)) for _ in range(n_tetes)]
    return {
        "tetes": tetes,
        "Wo": rng.normal(0, s, (d, d)),
        "W1": rng.normal(0, s, (d, 4 * d)), "b1": np.zeros(4 * d),
        "W2": rng.normal(0, s, (4 * d, d)), "b2": np.zeros(d),
    }


def bloc_transformer(X, p):
    """Un bloc (variante de GPT-2) :

        X = X + attention(layer_norm(X))     # les tokens se parlent
        X = X + feed_forward(layer_norm(X))  # chaque token réfléchit seul

    Entrée et sortie : (T, d), même forme.
    """
    _, a = attention_multi_tetes(layer_norm(X), p["tetes"], p["Wo"])
    X = X + a
    return X + feed_forward(layer_norm(X), p["W1"], p["b1"], p["W2"], p["b2"])
