"""Rétropropagation et descente de gradient pour DenseNetwork (forward.py).

Le détail pas à pas du calcul est expliqué dans scripts/a_la_main/05_retropropagation.py.
Convention : réseau [n_entrées, n_cachée, n_sortie], activations au choix pour
la couche cachée (clés de ACTIVATIONS) et "sigmoid" en sortie, perte MSE.
"""
import numpy as np

from activations import ACTIVATIONS
from forward import mse


def retropropagation(net, X, y):
    """Retourne la perte MSE et les gradients (dW1, db1, dW2, db2)."""
    f1, grad1 = ACTIVATIONS[net.activations[0]]
    f2, grad2 = ACTIVATIONS[net.activations[1]]

    z1 = X @ net.W[0] + net.b[0]
    a1 = f1(z1)
    z2 = a1 @ net.W[1] + net.b[1]
    a2 = f2(z2)

    dA2 = 2 * (a2 - y) / len(X)
    dZ2 = dA2 * grad2(z2)
    dW2 = a1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ net.W[1].T) * grad1(z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return mse(a2, y), (dW1, db1, dW2, db2)


def descente(net, gradients, taux):
    """Un pas de descente de gradient : poids = poids - taux * gradient."""
    dW1, db1, dW2, db2 = gradients
    net.W[0] -= taux * dW1
    net.b[0] -= taux * db1
    net.W[1] -= taux * dW2
    net.b[1] -= taux * db2
