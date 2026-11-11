"""Optimiseur Adam pour DenseNetwork (forward.py), écrit à la main.

Adam garde, pour chaque paramètre, deux moyennes mobiles du gradient g :
    m = moyenne du gradient          (la "direction", lisse comme de l'élan)
    v = moyenne du gradient au carré (l'"échelle" typique du gradient)
et déplace le paramètre de  taux * m / (racine(v) + eps).
Les divisions par (1 - beta^t) corrigent le biais des premières étapes.
"""
import numpy as np


class Adam:
    def __init__(self, net, taux=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = [net.W[0], net.b[0], net.W[1], net.b[1]]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.taux, self.beta1, self.beta2, self.eps = taux, beta1, beta2, eps
        self.t = 0

    def pas(self, gradients):
        """Un pas d'Adam. gradients = (dW1, db1, dW2, db2), comme retropropagation()."""
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_corrige = self.m[i] / (1 - self.beta1 ** self.t)
            v_corrige = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.taux * m_corrige / (np.sqrt(v_corrige) + self.eps)
