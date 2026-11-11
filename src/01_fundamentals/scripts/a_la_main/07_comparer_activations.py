"""
LEÇON 7 : quelle activation choisir pour le XOR ?
=================================================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/07_comparer_activations.py

PLACE DANS LE PARCOURS
Même exemple et même entraînement qu'à la leçon 6 (XOR, réseau [2, 4, 1], MSE).
On change UNE SEULE chose : l'activation de la couche cachée (leçon 2).

QUESTION
Quelle activation est la meilleure pour notre problème ? On juge sur deux critères :
  1. le moins de neurones "morts" (un neurone mort ne sert plus à rien),
  2. la convergence la plus rapide (le moins d'époques pour apprendre).

MÉTHODE
Le résultat dépend des poids de départ (aléatoires). On refait donc l'entraînement
avec 10 départs différents (graines 0 à 9) pour chaque activation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès aux modules
from backpropagation import descente, retropropagation
from forward import DenseNetwork

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

TAUX = 0.5            # mêmes réglages que la leçon 6, pour toutes les activations
SEUIL_PERTE = 0.01
MAX_EPOQUES = 20000
GRAINES = range(10)


def entrainer(activation, graine):
    """Retourne (époques nécessaires ou None si échec, nombre de neurones morts)."""
    net = DenseNetwork([2, 4, 1], [activation, "sigmoid"], seed=graine)
    epoques = None
    for epoque in range(MAX_EPOQUES + 1):
        perte, gradients = retropropagation(net, X, y)
        if perte < SEUIL_PERTE:
            epoques = epoque
            break
        descente(net, gradients, TAUX)
    # Neurone "mort" : Z <= 0 pour les 4 exemples (relu et leaky_relu seulement).
    z1 = X @ net.W[0] + net.b[0]
    morts = int(np.sum(np.all(z1 <= 0, axis=0)))
    return epoques, morts


# ---------------------------------------------------------------
# ÉTAPE 1 : mesurer
# ---------------------------------------------------------------
print("ÉTAPE 1 - 10 entraînements par activation")
print(f"  {'activation':<11} {'réussites':>9} {'époques (médiane)':>18} {'neurones morts (moyenne)':>26}")
for activation in ["tanh", "relu", "leaky_relu", "sigmoid"]:
    essais = [entrainer(activation, g) for g in GRAINES]
    epoques = [e for e, _ in essais if e is not None]
    if activation in ("relu", "leaky_relu"):
        morts = f"{np.mean([m for _, m in essais]):.1f}"
    else:
        morts = "0 (ne meurt pas)"  # tanh et sigmoid ont toujours une pente non nulle
    print(f"  {activation:<11} {len(epoques):>6}/{len(essais)} {int(np.median(epoques)):>18} {morts:>26}")


# ---------------------------------------------------------------
# ÉTAPE 2 : le verdict pour le XOR
# ---------------------------------------------------------------
# tanh       : réussit toujours, pas de neurone mort, ~320 époques  -> LA MEILLEURE.
# relu       : aussi rapide (~280) mais 1,5 neurone mort par réseau en moyenne :
#              3 essais sur 10 échouent, le réseau bloqué sans assez de neurones.
# leaky_relu : réussit toujours, mais des neurones presque morts : certains
#              essais prennent 4000 à 6800 époques.
# sigmoid    : réussit toujours, pas de neurone mort, mais ~6 fois plus lente
#              (~1840 époques) : sa pente est <= 0.25 (leçon 2).
# Neurone mort : relu(z) = 0 et sa dérivée aussi pour z <= 0. Un neurone dont Z est
# négatif pour tous les exemples ne reçoit plus de gradient et ne change plus.
print("\nÉTAPE 2 - verdict pour le XOR : tanh (aucun échec, aucun neurone mort, ~320 époques)")
