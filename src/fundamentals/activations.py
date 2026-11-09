"""Fonctions d'activation et leurs dérivées (nécessaires à la rétropropagation)."""
import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.linspace(-5, 5, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, (f, g) in ACTIVATIONS.items():
        ax1.plot(z, f(z), label=name)
        ax2.plot(z, g(z), label=name)
    ax1.set_title("Activations")
    ax2.set_title("Dérivées")
    ax1.set_ylim(-2, 5)
    for ax in (ax1, ax2):
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend()
    plt.savefig("logs/activations.png")
    print("Courbes enregistrées dans logs/activations.png")
