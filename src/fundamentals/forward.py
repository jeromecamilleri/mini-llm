"""Petit réseau dense en NumPy : propagation avant uniquement.

Convention : X de forme (n_exemples, n_entrées) ; chaque couche calcule
Z = A_prev @ W + b puis A = activation(Z).
"""
import numpy as np

from activations import ACTIVATIONS


class DenseNetwork:
    def __init__(self, sizes, activations, seed=0):
        """sizes=[2, 4, 1] -> 2 entrées, 1 couche cachée de 4, 1 sortie."""
        assert len(activations) == len(sizes) - 1
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
                  for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in sizes[1:]]
        self.activations = activations

    def forward(self, X, verbose=False):
        a = X
        for i, (W, b, name) in enumerate(zip(self.W, self.b, self.activations)):
            z = a @ W + b
            a = ACTIVATIONS[name][0](z)
            if verbose:
                print(f"Couche {i + 1}: {name:10s} Z{z.shape} -> A{a.shape}")
        return a


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


if __name__ == "__main__":
    net = DenseNetwork([2, 4, 1], ["tanh", "sigmoid"])
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

    y_pred = net.forward(X, verbose=True)
    print("Sorties :", y_pred.ravel().round(4))
    print("Perte MSE (réseau non entraîné) :", round(mse(y_pred, y), 4))

    # Vérification manuelle de la première couche pour le 1er exemple
    z1 = X[0] @ net.W[0] + net.b[0]
    print("Z1 manuel :", z1.round(4), "| A1 :", np.tanh(z1).round(4))
