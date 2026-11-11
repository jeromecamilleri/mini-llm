"""GPT en PyTorch : la même architecture que 03_transformer/tools/gpt.py, mais avec autograd.

Détail pas à pas dans scripts/01_gpt_pytorch.py.
Pourquoi PyTorch ? Pour que les gradients (donc l'apprentissage) soient calculés automatiquement.
"""
import math

import torch
from torch import nn


class Attention(nn.Module):
    """Attention causale à plusieurs têtes (leçons 2, 3 et 5 de 03_transformer)."""

    def __init__(self, d, n_tetes):
        super().__init__()
        self.n_tetes, self.d_tete = n_tetes, d // n_tetes
        self.qkv = nn.Linear(d, 3 * d)  # Wq, Wk, Wv de toutes les têtes d'un coup
        self.Wo = nn.Linear(d, d)

    def forward(self, X):  # X : (B, T, d)
        B, T, d = X.shape
        Q, K, V = self.qkv(X).split(d, dim=-1)
        # (B, T, d) -> (B, têtes, T, d_tete)
        Q, K, V = (m.view(B, T, self.n_tetes, self.d_tete).transpose(1, 2) for m in (Q, K, V))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_tete)
        masque = torch.tril(torch.ones(T, T, dtype=torch.bool, device=X.device))
        scores = scores.masked_fill(~masque, float("-inf"))
        sortie = scores.softmax(dim=-1) @ V
        return self.Wo(sortie.transpose(1, 2).reshape(B, T, d))


class Bloc(nn.Module):
    """X = X + attention(LN(X)) ; X = X + feed_forward(LN(X)) (leçon 6 de 03_transformer)."""

    def __init__(self, d, n_tetes):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = Attention(d, n_tetes)
        # GELU « tanh » : même formule que 03_transformer (et GPT-2)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, X):
        X = X + self.attention(self.ln1(X))
        return X + self.ff(self.ln2(X))


class GPT(nn.Module):
    """ids (B, T) -> scores (B, T, taille_vocab)."""

    def __init__(self, taille_vocab, d, n_tetes, n_blocs, T_max):
        super().__init__()
        self.T_max = T_max
        self.emb_mots = nn.Embedding(taille_vocab, d)
        self.emb_pos = nn.Embedding(T_max, d)
        self.blocs = nn.ModuleList(Bloc(d, n_tetes) for _ in range(n_blocs))
        self.ln_final = nn.LayerNorm(d)
        self.sortie = nn.Linear(d, taille_vocab, bias=False)

    def forward(self, ids):
        T = ids.shape[1]
        X = self.emb_mots(ids) + self.emb_pos(torch.arange(T, device=ids.device))
        for bloc in self.blocs:
            X = bloc(X)
        return self.sortie(self.ln_final(X))

    def nb_parametres(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generer(self, ids, n_nouveaux, temperature=0.0, generateur=None):
        """Ajoute n_nouveaux tokens. temperature=0 : glouton ; sinon tirage (leçon 7 de 03_transformer)."""
        ids = ids.clone()
        for _ in range(n_nouveaux):
            scores = self(ids[:, -self.T_max:])[:, -1]  # dernière position : (B, V)
            if temperature == 0:
                suivant = scores.argmax(dim=-1, keepdim=True)
            else:
                probas = (scores / temperature).softmax(dim=-1)
                suivant = torch.multinomial(probas, 1, generator=generateur)
            ids = torch.cat([ids, suivant], dim=1)
        return ids


def perte(modele, x, y):
    """Entropie croisée moyenne sur toutes les positions du batch."""
    scores = modele(x)  # (B, T, V)
    return nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), y.reshape(-1))


@torch.no_grad()
def depuis_numpy(modele, p):
    """Copie dans le GPT PyTorch les poids du GPT NumPy (03_transformer/tools/gpt.py).

    Sert à vérifier que les deux versions calculent la même chose. Un Linear PyTorch
    stocke sa matrice transposée (sortie, entrée), d'où les .T. Les biais de l'attention
    n'existent pas en NumPy : on les met à zéro.
    """
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    modele.emb_mots.weight.copy_(t(p["E"]))
    modele.emb_pos.weight.copy_(t(p["P"]))
    modele.sortie.weight.copy_(t(p["W_sortie"].T))
    for bloc, b in zip(modele.blocs, p["blocs"]):
        Wq = torch.cat([t(w[0]) for w in b["tetes"]], dim=1)  # têtes côte à côte : (d, d)
        Wk = torch.cat([t(w[1]) for w in b["tetes"]], dim=1)
        Wv = torch.cat([t(w[2]) for w in b["tetes"]], dim=1)
        bloc.attention.qkv.weight.copy_(torch.cat([Wq, Wk, Wv], dim=1).T)
        bloc.attention.qkv.bias.zero_()
        bloc.attention.Wo.weight.copy_(t(b["Wo"].T))
        bloc.attention.Wo.bias.zero_()
        bloc.ff[0].weight.copy_(t(b["W1"].T))
        bloc.ff[0].bias.copy_(t(b["b1"]))
        bloc.ff[2].weight.copy_(t(b["W2"].T))
        bloc.ff[2].bias.copy_(t(b["b2"]))
