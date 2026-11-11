"""Chapitre 01_fundamentals : un neurone artificiel avec NumPy.

Un neurone calcule une somme pondérée des entrées plus un biais,
puis applique une fonction d'activation : y = activation(w . x + b)
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def neuron(x, w, b):
    """Propagation avant d'un neurone."""
    return sigmoid(np.dot(w, x) + b)


def mse(y_pred, y_true):
    """Fonction de perte : erreur quadratique moyenne."""
    return np.mean((y_pred - y_true) ** 2)


if __name__ == "__main__":
    x = np.array([1.0, 2.0])

    print("Effet des poids et du biais sur la sortie")
    for w, b in [([0.5, -0.5], 0.0), ([1.0, 1.0], 0.0), ([1.0, 1.0], -3.0), ([-1.0, -1.0], 3.0)]:
        print(f"w={w}, b={b:+.1f} -> y={neuron(x, np.array(w), b):.4f}")

    # Perte en fonction du poids w1 pour une cible fixée (w2 et b constants)
    target = 0.9
    w1_values = np.linspace(-4, 4, 200)
    losses = [mse(neuron(x, np.array([w1, 0.5]), 0.0), target) for w1 in w1_values]

    plt.plot(w1_values, losses)
    plt.xlabel("w1")
    plt.ylabel("perte (MSE)")
    plt.title("Perte en fonction du poids w1")
    plt.savefig("logs/perceptron_loss.png")
    print("Courbe enregistrée dans logs/perceptron_loss.png")
