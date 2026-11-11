"""Outils de l'attention (chapitre 03_transformer)."""
import numpy as np


def softmax(scores):
    """Transforme des scores en poids positifs de somme 1 (sur la dernière dimension).

    On retire le maximum avant l'exponentielle : le résultat est le même,
    mais on évite les nombres trop grands.
    """
    scores = np.asarray(scores, dtype=float)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention_simple(E):
    """Attention la plus simple : chaque token se mélange avec tous les autres.

    E : (T, d), un vecteur par token.
    Retourne (scores, poids, sortie) :
      scores = E @ E.T          (T, T)  ressemblance de chaque paire de tokens
      poids  = softmax(scores)  (T, T)  chaque ligne a pour somme 1
      sortie = poids @ E        (T, d)  moyenne pondérée des vecteurs
    """
    scores = E @ E.T
    poids = softmax(scores)
    return scores, poids, poids @ E


def masque_causal(T):
    """Matrice (T, T) de booléens : True là où un token a le DROIT de regarder.

    Le token i regarde les tokens 0..i (lui-même et le passé), jamais le futur.
    """
    return np.tril(np.ones((T, T), dtype=bool))


def attention_causale(E):
    """Comme attention_simple, mais sans regarder le futur.

    Les scores interdits sont remplacés par -infini : après softmax, leur poids vaut 0.
    """
    scores = E @ E.T
    scores_masques = np.where(masque_causal(len(E)), scores, -np.inf)
    poids = softmax(scores_masques)
    return scores_masques, poids, poids @ E


def attention_qkv(E, Wq, Wk, Wv, causal=True):
    """Attention avec Query, Key, Value (une seule tête).

    E : (T, d). Wq, Wk : (d, dk). Wv : (d, dv).
      Q = E @ Wq   ce que chaque token CHERCHE
      K = E @ Wk   l'étiquette que chaque token PRÉSENTE
      V = E @ Wv   ce que chaque token APPORTE s'il est choisi
      scores = Q @ K.T / sqrt(dk)
    Retourne (Q, K, V, scores, poids, sortie).
    """
    Q, K, V = E @ Wq, E @ Wk, E @ Wv
    scores = Q @ K.T / np.sqrt(K.shape[1])
    if causal:
        scores = np.where(masque_causal(len(E)), scores, -np.inf)
    poids = softmax(scores)
    return Q, K, V, scores, poids, poids @ V


def ajouter_positions(E, P):
    """Ajoute à chaque token le vecteur de SA position : X[i] = E[i] + P[i].

    E : (T, d) vecteurs des tokens. P : (T_max, d) table des positions, T <= T_max.
    """
    return E + P[:len(E)]


def attention_multi_tetes(E, tetes, Wo):
    """Plusieurs têtes d'attention en parallèle, puis mélange final.

    E : (T, d). tetes : liste de (Wq, Wk, Wv), une par tête (chacune (d, d_tete)).
    Wo : (n_tetes * d_tete, d). Chaque tête a ses propres matrices, donc sa propre question.
      sorties des têtes -> mises bout à bout (concaténation) -> @ Wo
    Retourne (liste des poids par tête, sortie (T, d)).
    """
    poids, sorties = [], []
    for Wq, Wk, Wv in tetes:
        *_, p, o = attention_qkv(E, Wq, Wk, Wv)
        poids.append(p)
        sorties.append(o)
    return poids, np.concatenate(sorties, axis=1) @ Wo
