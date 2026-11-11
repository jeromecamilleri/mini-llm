"""
LEÇON 6 : entraîner le réseau sur le XOR
========================================

    .venv/bin/python src/01_fundamentals/scripts/a_la_main/06_entrainement.py

PLACE DANS LE PARCOURS
Leçon 3 : calculer la sortie. Leçon 5 : calculer les gradients (un pas).
Ici on RÉPÈTE ce pas : c'est l'ENTRAÎNEMENT. On regarde les poids bouger,
on décide quand s'arrêter, puis on utilise le réseau sur le XOR.

IDÉE CENTRALE
Une ÉPOQUE = un passage sur les 4 exemples :
    1. propagation avant + perte
    2. rétropropagation -> gradients
    3. descente : poids = poids - taux * gradient
On recommence jusqu'à un critère d'arrêt.

Le code de l'étape 2 est dans tools/backpropagation.py (leçon 05 pour le détail).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès aux modules
from backpropagation import descente, retropropagation
from forward import DenseNetwork

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR
net = DenseNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=0)

TAUX = 0.5            # taille du pas
SEUIL_PERTE = 0.01    # critère d'arrêt : "assez bon"
MAX_EPOQUES = 20000   # garde-fou : on s'arrête de toute façon ici


# ---------------------------------------------------------------
# ÉTAPE 1 : la boucle d'entraînement, avec les poids qui bougent
# ---------------------------------------------------------------
print("ÉTAPE 1 - entraînement (poids suivis : W1[0,0], W1[1,0], W2[0,0], b2)")
print(f"  {'époque':>6} | {'perte':>7} | {'W1[0,0]':>8} {'W1[1,0]':>8} {'W2[0,0]':>8} {'b2':>8}")

AFFICHER = {0, 10, 25, 50, 100, 200, 300}
perte = None
for epoque in range(MAX_EPOQUES + 1):
    perte, gradients = retropropagation(net, X, y)

    if epoque in AFFICHER or perte < SEUIL_PERTE:
        print(f"  {epoque:>6} | {perte:7.4f} | {net.W[0][0, 0]:8.3f} {net.W[0][1, 0]:8.3f}"
              f" {net.W[1][0, 0]:8.3f} {net.b[1][0]:8.3f}")

    # ---------------------------------------------------------------
    # ÉTAPE 2 : quand s'arrêter ?
    # ---------------------------------------------------------------
    if perte < SEUIL_PERTE:
        print(f"\nÉTAPE 2 - arrêt à l'époque {epoque} : perte {perte:.4f} < seuil {SEUIL_PERTE}")
        break
    descente(net, gradients, TAUX)
else:
    print(f"\nÉTAPE 2 - garde-fou : {MAX_EPOQUES} époques atteintes, perte = {perte:.4f}")
# Sans critère d'arrêt, on tournerait indéfiniment : la perte baisse de moins
# en moins et n'atteint jamais exactement 0.
# Trois critères courants :
#   - perte sous un seuil (ici, 0.01)
#   - nombre maximal d'époques (garde-fou)
#   - la perte ne baisse plus (stagnation)
# Sur un gros modèle, on surveille aussi la perte sur des données NON vues à
# l'entraînement, pour éviter le sur-apprentissage (leçon à venir).


# ---------------------------------------------------------------
# ÉTAPE 3 : utiliser le réseau entraîné
# ---------------------------------------------------------------
# On n'a plus besoin de la rétropropagation : seule la propagation avant sert.
def xor(x1, x2):
    sortie = net.forward(np.array([[x1, x2]]))[0, 0]
    return sortie, int(sortie > 0.5)  # au-dessus de 0.5 : on répond "1"


print("\nÉTAPE 3 - le réseau répond-il au XOR ?")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    sortie, reponse = xor(x1, x2)
    attendu = x1 ^ x2
    verdict = "OK" if reponse == attendu else "FAUX"
    print(f"  XOR({x1}, {x2}) : sortie = {sortie:.3f} -> réponse {reponse} (attendu {attendu}) {verdict}")
# La sortie est un nombre entre 0 et 1, jamais exactement 0 ou 1.
# On la transforme en réponse avec un seuil à 0.5.

print("\n  Et pour des entrées jamais vues (entre 0 et 1) ?")
for x1, x2 in [(0.9, 0.1), (0.1, 0.1), (0.5, 0.5)]:
    sortie, reponse = xor(x1, x2)
    print(f"  XOR({x1}, {x2}) : sortie = {sortie:.3f} -> réponse {reponse}")
# Le réseau n'a vu que 4 cas : hors de ces cas, il extrapole à sa façon.



# ---------------------------------------------------------------
# RÉFLEXION : un `if` ou un réseau entraîné ?
# ---------------------------------------------------------------
# On aurait pu écrire : if x1 != x2: return 1. Ici, AUCUNE règle n'a été écrite :
# 4 exemples ont suffi pour que le réseau trouve seul ses 17 paramètres.
# Pour le XOR, le `if` est plus simple ; l'intérêt est ailleurs : quand personne
# ne sait écrire la règle (image, traduction, mot suivant). On programme alors
# comment APPRENDRE, et les données façonnent les poids. C'est le principe d'un GPT.
