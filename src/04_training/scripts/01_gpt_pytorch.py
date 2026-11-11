"""
LEÇON 1 (training) : le GPT en PyTorch et sa première perte
===========================================================

    .venv/bin/python src/04_training/scripts/01_gpt_pytorch.py

PLACE DANS LE PARCOURS (chapitre 04_training)
Le GPT de 03_transformer était écrit en NumPy : parfait pour voir la mécanique,
mais NumPy ne sait pas calculer les GRADIENTS. Or entraîner = corriger les poids
dans le sens indiqué par le gradient (01_fundamentals, partie PyTorch). On réécrit
donc EXACTEMENT la même architecture en PyTorch (tools/gpt_torch.py) : embeddings
+ positions, blocs (attention causale, feed forward, résiduelles, LayerNorm), couche
finale. Seule différence : PyTorch calcule les gradients tout seul (autograd).

L'ESSENTIEL
  1. même architecture, même forme : (B, T) -> (B, T, d) -> (B, T, vocabulaire) ;
  2. une PERTE (cross entropy) mesure l'erreur : -log(probabilité du vrai mot suivant) ;
  3. avant tout apprentissage, la perte vaut environ ln(V) ;
  4. loss.backward() donne un gradient pour CHAQUE paramètre : c'est lui qui
     dira, à la leçon 2, dans quel sens corriger.

B = taille du batch (nombre de fenêtres traitées ensemble), T = longueur d'une fenêtre.

POURQUOI PASSER À PYTORCH DÈS MAINTENANT ?
Entraîner demande le gradient de TOUS les paramètres (têtes, feed forward,
LayerNorm, embeddings). En NumPy il faudrait écrire la rétropropagation à la main
pour chaque couche (softmax, masque, LayerNorm, GELU...) : long et facile à rater.
En 01_fundamentals, on l'a faite à la main sur de petits réseaux : on en connaît le
principe, et backward() fait exactement la même chose automatiquement.
Ce qui ne change PAS : l'architecture, et les mécanismes (attention, masque,
softmax) restent écrits à la main. On n'utilise que des briques de base
(Linear, Embedding, LayerNorm, autograd), jamais un GPT préfabriqué.
Pour s'en convaincre : l'ÉTAPE 2 vérifie que la version PyTorch donne les MÊMES
scores que la version NumPy de 03_transformer avec les mêmes poids.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_transformer" / "tools"))
from donnees import charger_corpus, tirer_batch
from gpt import gpt as gpt_numpy, init_gpt  # version NumPy (03_transformer)
from gpt_torch import GPT, depuis_numpy, perte

torch.manual_seed(0)
tok, entrainement, validation = charger_corpus()
V = len(tok.vocab)
T = 4  # fenêtre de contexte (un vrai GPT : 256 à plusieurs milliers)
modele = GPT(taille_vocab=V, d=32, n_tetes=2, n_blocs=2, T_max=T)

print("ÉTAPE 1 : la taille du modèle")
print(f"  vocabulaire V = {V}, d = 32, 2 têtes, 2 blocs, T = {T}")
print(f"  paramètres : {modele.nb_parametres()}")
for nom, p in modele.named_parameters():
    if nom.startswith(("emb", "sortie", "blocs.0")):
        print(f"    {nom:<28} {tuple(p.shape)}")

print("\nÉTAPE 2 : vérification, PyTorch contre NumPy (mêmes poids)")
p_numpy = init_gpt(V, d=8, n_tetes=2, n_blocs=2, T_max=T, rng=np.random.default_rng(0))
petit = GPT(taille_vocab=V, d=8, n_tetes=2, n_blocs=2, T_max=T)
depuis_numpy(petit, p_numpy)
ids = [10, 4, 8, 2]  # le chat dort .
sortie_numpy = gpt_numpy(ids, p_numpy)
with torch.no_grad():
    sortie_torch = petit(torch.tensor([ids]))[0].numpy()
ecart = np.abs(sortie_numpy - sortie_torch).max()
print(f"  scores NumPy {sortie_numpy.shape}, scores PyTorch {sortie_torch.shape}")
print(f"  écart maximal entre les deux : {ecart:.1e}")
print("  Écart ~1e-4 : float32 (PyTorch) contre float64 (NumPy), et un epsilon de LayerNorm écrit")
print("  un peu différemment. Les scores sont les mêmes : seule la façon de l'écrire change.")

print("\nÉTAPE 3 : un batch traverse le modèle")
g = torch.Generator().manual_seed(0)
x, y = tirer_batch(entrainement, 8, T, g)
scores = modele(x)
print(f"  x : {tuple(x.shape)}  ->  scores : {tuple(scores.shape)}   (B, T, V)")

print("\nÉTAPE 4 : la perte avant tout apprentissage")
l = perte(modele, x, y)
print(f"  perte = {l.item():.3f}   ln(V) = ln({V}) = {np.log(V):.3f}")
print("  Le modèle hésite à peu près entre tous les mots : c'est le POINT DE DÉPART.")
# Cross entropy = -log(proba donnée au VRAI mot). Si la proba vaut 1/V, on obtient ln(V).

print("\nÉTAPE 5 : le gradient (autograd)")
l.backward()
grad = modele.sortie.weight.grad
print(f"  modele.sortie.weight.grad existe : forme {tuple(grad.shape)}, norme {grad.norm():.3f}")
print("  Chaque paramètre a maintenant un gradient : « dans quel sens bouger pour réduire la perte ».")

print("\nÉTAPE 6 : un tout petit pas dans le sens du gradient")
with torch.no_grad():
    for p in modele.parameters():
        p -= 0.1 * p.grad
print(f"  perte avant : {l.item():.3f}   perte après : {perte(modele, x, y).item():.3f}")
print("  Elle baisse (sur CE batch) : entraîner = répéter ce pas des milliers de fois. -> leçon 2.")

print("\nÉTAPE 7 : le masque causal fonctionne encore")
modele.eval()
with torch.no_grad():
    a = modele(x)
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % V  # on change le DERNIER mot de chaque fenêtre
    b = modele(x2)
print("  scores du passé inchangés quand on modifie le dernier mot :",
      torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6))
