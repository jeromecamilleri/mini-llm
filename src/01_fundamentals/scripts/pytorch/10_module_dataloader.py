"""
LEÇON 10 : tenseurs, nn.Module et DataLoader
============================================

    .venv/bin/python src/01_fundamentals/scripts/pytorch/10_module_dataloader.py

PLACE DANS LE PARCOURS (partie PyTorch de 01_fundamentals)
Même exemple : le XOR, réseau [2, 4, 1] (tanh, sigmoid), perte MSE.
Leçon 9 : PyTorch calcule les gradients, mais nos 4 paramètres traînaient en
vrac dans une liste. Ici on range le réseau comme dans tout vrai projet PyTorch :
  - les formes (dimensions) des tenseurs, à chaque étape ;
  - une classe nn.Module qui contient les paramètres et la propagation avant ;
  - un DataLoader qui distribue les exemples par petits paquets (« batchs »).
C'est exactement la structure qu'aura notre mini-GPT.

On n'utilise pas nn.Linear : on déclare W et b nous-mêmes (nn.Parameter),
pour garder la main sur ce que fait le réseau.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Le XOR est trop petit pour la GPU (leçon 9) : on reste sur le CPU.
# Pour un gros réseau, on mettrait "cuda" ici et tout le reste suivrait.
appareil = "cpu"
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR


# ---------------------------------------------------------------
# ÉTAPE 1 : les formes des tenseurs
# ---------------------------------------------------------------
# Règle : (N, entrées) @ (entrées, sorties) -> (N, sorties), N = nb d'exemples.
# 90 % des bugs PyTorch sont des formes incompatibles : on les regarde.
torch.manual_seed(0)
W1, b1 = torch.randn(2, 4), torch.zeros(4)
W2, b2 = torch.randn(4, 1), torch.zeros(1)
Z1 = X @ W1 + b1
A1 = torch.tanh(Z1)
Z2 = A1 @ W2 + b2
print("ÉTAPE 1 - formes :")
for nom, t in [("X", X), ("W1", W1), ("Z1 = X @ W1 + b1", Z1), ("A1 = tanh(Z1)", A1),
               ("W2", W2), ("Z2 = A1 @ W2 + b2", Z2)]:
    print(f"         {nom:20s} {tuple(t.shape)}")
# b1 (4,) est ajouté à chaque ligne de X @ W1 (4, 4) : c'est le « broadcast ».
print("         dtype :", X.dtype, "| appareil :", X.device)
# float32 : 4 octets par nombre, le format normal des réseaux (leçon 9 : float64).


# ---------------------------------------------------------------
# ÉTAPE 2 : une classe nn.Module
# ---------------------------------------------------------------
# nn.Parameter = un tenseur que PyTorch sait être « à apprendre » :
# il a requires_grad=True et apparaît dans reseau.parameters().
class ReseauXor(nn.Module):
    def __init__(self, n_entrees=2, n_cachee=4, n_sortie=1):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(n_entrees, n_cachee) * (2 / n_entrees) ** 0.5)  # init He, comme DenseNetwork
        self.b1 = nn.Parameter(torch.zeros(n_cachee))
        self.W2 = nn.Parameter(torch.randn(n_cachee, n_sortie) * (2 / n_cachee) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_sortie))

    def forward(self, x):  # la propagation avant (leçon 3)
        a1 = torch.tanh(x @ self.W1 + self.b1)
        return torch.sigmoid(a1 @ self.W2 + self.b2)


torch.manual_seed(0)
reseau = ReseauXor().to(appareil)  # .to("cuda") enverrait TOUS les paramètres sur la GPU
print("\nÉTAPE 2 - paramètres du réseau :")
for nom, p in reseau.named_parameters():
    print(f"         {nom:3s} {str(tuple(p.shape)):7s} {p.numel()} nombres")
print("         total :", sum(p.numel() for p in reseau.parameters()), "(les 17 de la leçon 6)")
# reseau(x) appelle reseau.forward(x) : on s'en sert comme d'une fonction.


# ---------------------------------------------------------------
# ÉTAPE 3 : Dataset et DataLoader
# ---------------------------------------------------------------
# Un Dataset associe chaque entrée à sa cible. Le DataLoader les distribue
# par paquets (batchs), dans un ordre mélangé à chaque époque (shuffle).
# Avec 4 exemples c'est inutile ; avec des millions de textes, indispensable :
# on ne peut pas tout mettre dans la mémoire de la GTX 1070 d'un coup.
donnees = TensorDataset(X, y)
print("\nÉTAPE 3 - un passage (une époque) avec batchs de 2 :")
generateur = torch.Generator().manual_seed(0)
for i, (xb, yb) in enumerate(DataLoader(donnees, batch_size=2, shuffle=True, generator=generateur)):
    print(f"         batch {i} : x {tuple(xb.shape)}  entrées {xb.tolist()}  cibles {yb.ravel().tolist()}")


# ---------------------------------------------------------------
# ÉTAPE 4 : la boucle d'entraînement complète
# ---------------------------------------------------------------
# Mêmes 4 gestes que la leçon 9, mais pour chaque batch : zero_grad, perte,
# backward, step. Une ÉPOQUE = tous les batchs vus une fois.
def entrainer(batch_size, graine=0, taux=0.05, seuil=0.01, max_epoques=500):
    torch.manual_seed(graine)
    net = ReseauXor().to(appareil)
    opt = torch.optim.Adam(net.parameters(), lr=taux)
    chargeur = DataLoader(donnees, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(graine))
    for epoque in range(1, max_epoques + 1):
        for xb, yb in chargeur:
            xb, yb = xb.to(appareil), yb.to(appareil)
            opt.zero_grad()
            perte = ((net(xb) - yb) ** 2).mean()
            perte.backward()
            opt.step()
        with torch.no_grad():
            perte_totale = ((net(X.to(appareil)) - y.to(appareil)) ** 2).mean().item()
        if perte_totale < seuil:
            return net, epoque
    return net, None


print("\nÉTAPE 4 - époques pour passer sous 0.01 (Adam 0.05, graines 0 à 9) :")
print("         batch  réussites  médiane  pire")
for taille in [4, 2, 1]:
    resultats = [entrainer(taille, graine)[1] for graine in range(10)]
    ok = sorted(r for r in resultats if r is not None)
    print(f"         {taille:5d}  {len(ok):4d}/10  {ok[len(ok) // 2]:7d}  {ok[-1]:5d}")
# batch 4 = tout le jeu d'un coup (comme les leçons 6 à 9) : 1 correction par époque.
# batch 1 = 4 corrections par époque : moins d'époques, mais chacune coûte plus cher.

# Quelques graines échouent (poids de départ défavorables) : on prend la première qui réussit.
graine = next(g for g in range(10) if entrainer(4, g)[1])
net, epoque = entrainer(4, graine)
net.eval()  # mode « utilisation » (sans effet ici, mais indispensable avec dropout, leçon future)
with torch.no_grad():
    sortie = net(X.to(appareil)).cpu()
print(f"\n         réseau entraîné (graine {graine}, batch 4, {epoque} époques) :")
for entree, s in zip(X, sortie):
    print(f"          {int(entree[0])}  {int(entree[1])} ->  {s.item():.3f}")
