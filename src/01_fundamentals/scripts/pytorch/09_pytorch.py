"""
LEÇON 9 : PyTorch fait la rétropropagation à notre place
========================================================

    .venv/bin/python src/01_fundamentals/scripts/pytorch/09_pytorch.py

PLACE DANS LE PARCOURS
Même exemple : le XOR, réseau [2, 4, 1] (tanh, sigmoid), perte MSE.
Leçons 5 et 6 : on a écrit la rétropropagation À LA MAIN (NumPy).
Ici : PyTorch la calcule tout seul (« autograd »). On vérifie qu'il trouve
EXACTEMENT les mêmes 17 gradients que nous, puis on entraîne avec lui.

IDÉE CENTRALE
Un tenseur PyTorch avec requires_grad=True note chaque opération faite avec lui.
perte.backward() rejoue ces opérations à l'envers (la règle de dérivation en
chaîne de la leçon 5) et remplit .grad de chaque paramètre.
Pour la suite du projet (GPT), on ne codera plus les gradients à la main :
c'est autograd qui s'en chargera. Mais on sait maintenant ce qu'il fait.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))  # accès à forward.py
from backpropagation import retropropagation
from forward import DenseNetwork

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR
net = DenseNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=0)


# ---------------------------------------------------------------
# ÉTAPE 1 : le même réseau en PyTorch, avec les MÊMES poids
# ---------------------------------------------------------------
# float64 : même précision que NumPy, pour comparer au chiffre près.
def tenseur(a, grad=False):
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


Xt, yt = tenseur(X), tenseur(y)
parametres = [tenseur(net.W[0], True), tenseur(net.b[0], True),
              tenseur(net.W[1], True), tenseur(net.b[1], True)]


def perte_torch(W1, b1, W2, b2):
    a1 = torch.tanh(Xt @ W1 + b1)
    a2 = torch.sigmoid(a1 @ W2 + b2)
    return ((a2 - yt) ** 2).mean()


perte = perte_torch(*parametres)
print("ÉTAPE 1 - perte PyTorch :", round(perte.item(), 6))


# ---------------------------------------------------------------
# ÉTAPE 2 : autograd contre notre rétropropagation
# ---------------------------------------------------------------
perte.backward()  # remplit p.grad pour chacun des 4 paramètres
perte_main, grads_main = retropropagation(net, X, y)
print("\nÉTAPE 2 - écart entre nos gradients et ceux de PyTorch :")
for nom, p, g in zip(["dW1", "db1", "dW2", "db2"], parametres, grads_main):
    ecart = np.abs(p.grad.numpy() - g).max()
    print(f"         {nom} {str(tuple(g.shape)):7s} écart max = {ecart:.2e}")
# Les écarts sont de l'ordre de 1e-17 : ce sont des erreurs d'arrondi.
# Nos gradients faits main sont donc justes, pour les 17 paramètres.


# ---------------------------------------------------------------
# ÉTAPE 3 : entraîner avec PyTorch (autograd + Adam)
# ---------------------------------------------------------------
# Même boucle que la leçon 6 : calculer la perte, rétropropager, corriger.
# Ici backward() remplace notre retropropagation, et torch.optim.Adam
# remplace notre classe Adam de la leçon 8 (même formule).
for p in parametres:
    p.grad = None
optimiseur = torch.optim.Adam(parametres, lr=0.05)
for epoque in range(1, 20001):
    optimiseur.zero_grad()          # remet les gradients à zéro (sinon ils s'additionnent)
    perte = perte_torch(*parametres)
    if perte.item() < 0.01:
        break
    perte.backward()
    optimiseur.step()
print(f"\nÉTAPE 3 - PyTorch + Adam 0.05 : perte < 0.01 à l'époque {epoque}")
# Avec notre Adam (leçon 8), la graine 0 passait aussi sous le seuil vers
# l'époque 106 (c'est le cas le plus lent des 10 graines, médiane 58).

with torch.no_grad():  # pas besoin des gradients pour utiliser le réseau
    W1, b1, W2, b2 = parametres
    sortie = torch.sigmoid(torch.tanh(Xt @ W1 + b1) @ W2 + b2)
print("         x1 x2 -> sortie")
for entree, s in zip(X, sortie):
    print(f"          {int(entree[0])}  {int(entree[1])} ->  {s.item():.3f}")


# ---------------------------------------------------------------
# ÉTAPE 4 : la même chose sur la GPU
# ---------------------------------------------------------------
# .to("cuda") envoie un tenseur dans la mémoire de la GTX 1070 ; les calculs
# se font ensuite là-bas. Pour 17 paramètres c'est inutile (plus lent même),
# mais pour un GPT de millions de paramètres c'est indispensable.
if torch.cuda.is_available():
    W1g = parametres[0].detach().to("cuda")
    print("\nÉTAPE 4 - W1 est sur :", W1g.device, "| GPU :", torch.cuda.get_device_name(0))
else:
    print("\nÉTAPE 4 - pas de GPU disponible : tout reste sur le CPU")


# ---------------------------------------------------------------
# ÉTAPE 5 : CPU contre GPU, temps d'exécution
# ---------------------------------------------------------------
# Même boucle d'entraînement (perte, backward, Adam) répétée 500 fois,
# d'abord sur le XOR [2, 4, 1], puis sur un réseau 500 fois plus large.
import time


def duree(appareil, n_entrees, n_cachee, n_exemples, repetitions=500):
    g = torch.Generator().manual_seed(0)
    Xg = torch.rand(n_exemples, n_entrees, generator=g).to(appareil)
    yg = torch.rand(n_exemples, 1, generator=g).to(appareil)
    ps = [torch.randn(n_entrees, n_cachee, generator=g).to(appareil).requires_grad_(),
          torch.zeros(n_cachee, device=appareil, requires_grad=True),
          torch.randn(n_cachee, 1, generator=g).to(appareil).requires_grad_(),
          torch.zeros(1, device=appareil, requires_grad=True)]
    opt = torch.optim.Adam(ps, lr=0.01)
    if appareil == "cuda":
        torch.cuda.synchronize()  # la GPU travaille en différé : on attend la fin
    debut = time.perf_counter()
    for _ in range(repetitions):
        opt.zero_grad()
        W1, b1, W2, b2 = ps
        perte = ((torch.sigmoid(torch.tanh(Xg @ W1 + b1) @ W2 + b2) - yg) ** 2).mean()
        perte.backward()
        opt.step()
    if appareil == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - debut


if torch.cuda.is_available():
    duree("cuda", 2, 4, 4, 20)  # échauffement : le premier appel GPU est lent
    print("\nÉTAPE 5 - 500 pas d'entraînement (secondes)")
    print("         réseau                         CPU     GPU   plus rapide")
    for nom, args in [("XOR [2, 4, 1], 4 exemples ", (2, 4, 4)),
                      ("large [256, 2048, 1], 4096 ex.", (256, 2048, 4096))]:
        t_cpu, t_gpu = duree("cpu", *args), duree("cuda", *args)
        gagnant = f"CPU x{t_gpu / t_cpu:.1f}" if t_cpu < t_gpu else f"GPU x{t_cpu / t_gpu:.1f}"
        print(f"         {nom:30s} {t_cpu:6.2f}  {t_gpu:6.2f}   {gagnant}")
# Sur le XOR, la GPU est PLUS LENTE : chaque petite opération doit être
# lancée sur la carte (coût fixe) et le calcul lui-même est minuscule.
# Sur le gros réseau, des milliers de multiplications se font en parallèle
# et la GPU gagne. Un GPT est dans le second cas.
