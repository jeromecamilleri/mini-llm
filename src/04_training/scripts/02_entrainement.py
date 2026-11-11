"""
LEÇON 2 (training) : la boucle d'entraînement
=============================================

    .venv/bin/python src/04_training/scripts/02_entrainement.py

PLACE DANS LE PARCOURS (chapitre 04_training)
Leçon 1 : un GPT PyTorch dont la perte de départ vaut ln(V) (il hésite entre tous
les mots) et un gradient pour chaque paramètre. Ici on RÉPÈTE le petit pas des
milliers de fois : c'est l'entraînement.

L'ESSENTIEL : une boucle de 5 lignes (la même qu'en 01_fundamentals !)
  for etape in range(N):
      x, y = tirer_batch(...)          # des fenêtres au hasard
      l = perte(modele, x, y)          # à quel point on se trompe
      optimiseur.zero_grad()           # on efface les anciens gradients
      l.backward()                     # nouveaux gradients (autograd)
      optimiseur.step()                # AdamW corrige les poids

Ce qu'on a déjà vu : AdamW (version d'Adam qui « fait fondre » aussi un peu les
poids, ce qui limite le surapprentissage), perte d'entraînement contre perte de
VALIDATION (11_validation), surapprentissage.

SURAPPRENTISSAGE ICI
Le corpus ne contient que 67 mots : on a plus de paramètres que de données.
On verra donc la perte d'entraînement tomber très bas, alors que la validation
(7 mots jamais vus) reste mauvaise : le modèle apprend PAR CŒUR. Avec un vrai
corpus (datasets/), la validation suivra l'entraînement bien plus longtemps.

PERPLEXITÉ = exp(perte) : « parmi combien de mots hésite-t-on en moyenne ? ».
Au départ : exp(ln V) = V (tous les mots). Après entraînement : proche de 1 ou 2.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from donnees import charger_corpus, tirer_batch
from gpt_torch import GPT, perte

torch.manual_seed(0)
tok, entrainement, validation = charger_corpus()
V = len(tok.vocab)
T = 4
modele = GPT(taille_vocab=V, d=32, n_tetes=2, n_blocs=2, T_max=T)
optimiseur = torch.optim.AdamW(modele.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(0)


def moyenne_perte(donnees, n_batchs=20):
    """Perte moyenne sur des fenêtres tirées au hasard, SANS corriger les poids."""
    gen = torch.Generator().manual_seed(123)  # mêmes fenêtres à chaque mesure
    modele.eval()
    with torch.no_grad():
        total = sum(perte(modele, *tirer_batch(donnees, 16, T, gen)).item() for _ in range(n_batchs))
    modele.train()
    return total / n_batchs


def phrase(debut, n=10):
    ids = torch.tensor([tok.encoder(debut)])
    return tok.decoder(modele.generer(ids, n)[0].tolist())


print("ÉTAPE 1 : avant l'entraînement")
print(f"  perte train {moyenne_perte(entrainement):.3f}   val {moyenne_perte(validation):.3f}   ln(V) = {np.log(V):.3f}")
print(f"  génération : {phrase('le chat')}")

print("\nÉTAPE 2 : l'entraînement (AdamW, lr = 0,003, batch de 16 fenêtres)")
for etape in range(1, 601):
    x, y = tirer_batch(entrainement, 16, T, g)
    l = perte(modele, x, y)
    optimiseur.zero_grad()
    l.backward()
    optimiseur.step()
    if etape % 100 == 0:
        pt, pv = moyenne_perte(entrainement), moyenne_perte(validation)
        print(f"  étape {etape:>3} : train {pt:.3f}   val {pv:.3f}   perplexité train {np.exp(pt):.2f}")

print("\nÉTAPE 3 : générer avec le modèle entraîné")
for debut in ("le chat", "le cheval", "l ' oiseau"):
    print(f"  glouton  {debut!r:<14} -> {phrase(debut)}")
gen = torch.Generator().manual_seed(1)
ids = torch.tensor([tok.encoder("le chat")])
for t in (0.5, 1.0):
    sortie = modele.generer(ids, 10, temperature=t, generateur=gen)[0].tolist()
    print(f"  tirage, température {t} -> {tok.decoder(sortie)}")
print("  Le texte ressemble au corpus : le modèle a appris « sujet verbe . ».")

print("\nÉTAPE 4 : surapprentissage ?")
pt, pv = moyenne_perte(entrainement), moyenne_perte(validation)
print(f"  train {pt:.3f}  contre  val {pv:.3f}")
print("  Si val >> train : le modèle a surtout mémorisé ses 60 mots d'entraînement.")
