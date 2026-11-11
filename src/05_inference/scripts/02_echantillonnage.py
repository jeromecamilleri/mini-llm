"""
LEÇON 2 (inference) : température, Top-K, Top-P
================================================

    .venv/bin/python src/05_inference/scripts/02_echantillonnage.py

PLACE DANS LE PARCOURS (chapitre 05_inference)
Leçon 1 : à chaque tour, le modèle donne des probabilités ; on choisit un mot
(glouton ou tirage). Ici on RÈGLE le tirage. Trois réglages, qui agissent sur les
probabilités AVANT le tirage (les poids du modèle ne changent pas) :

  TEMPÉRATURE T : on divise les scores par T avant le softmax.
      T < 1 : le favori domine (proche du glouton) ; T > 1 : probabilités aplaties.
  TOP-K : on ne garde que les K mots les plus probables, on renormalise.
      (K = 3 : « on ne tire qu'entre les 3 favoris ».)
  TOP-P (« noyau ») : on garde les mots les plus probables jusqu'à cumuler p
      (p = 0,9 : « les mots qui font 90 % des chances à eux tous »).
      Différence avec Top-K : le NOMBRE de mots gardés s'adapte. Si le modèle est sûr
      de lui, 1 mot suffit ; s'il hésite, on en garde beaucoup.

« Renormaliser » : après avoir mis à 0 les mots écartés, on divise par la somme
restante pour que le total fasse de nouveau 100 %.

POURQUOI FILTRER ? Le tirage pur peut choisir un mot très improbable (rare, mais
après des centaines de mots ça arrive) et le texte déraille ensuite. Top-K et Top-P
coupent la « queue » de mots improbables.

Mesure utilisée ici : sur beaucoup de tirages après « le chat », quelle part des mots
choisis est un VRAI verbe (dort, mange, court) ? Et combien de mots différents sort-on ?

Écrit : fiches/02_echantillonnage.png.
"""
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generation import (charger_modele, choisir, filtrer_top_k, filtrer_top_p, probabilites,
                        scores_suivant)

tok, modele = charger_modele()
mots = tok.vocab.id_vers_token
ids = tok.encoder("le chat")
scores = scores_suivant(modele, ids)
verbes = {tok.vocab.token_vers_id[m] for m in ("dort", "mange", "court")}


def affiche(probas, n=5):
    haut = [i for i in probas.argsort(descending=True)[:n] if probas[i] > 0]
    return "  ".join(f"{mots[int(i)]} {probas[i].item():.1%}" for i in haut)


print("ÉTAPE 1 : la température change les probabilités (prompt « le chat »)")
for t in (0.5, 1.0, 2.0):
    print(f"  T = {t:<3} : {affiche(probabilites(scores, t))}")
print("  T basse : `dort` écrase les autres. T haute : tout devient plus proche, même les mots absurdes.")

print("\nÉTAPE 2 : Top-K et Top-P (T = 1)")
base = probabilites(scores)
print(f"  sans filtre  : {affiche(base)}")
print(f"  Top-K, K = 2 : {affiche(filtrer_top_k(base, 2))}")
print(f"  Top-P, p=0,8 : {affiche(filtrer_top_p(base, 0.8))}")
print("  Les mots écartés tombent à 0 % et les autres se partagent les 100 % (renormalisation).")

print("\nÉTAPE 3 : le nombre de mots gardés par Top-P s'adapte, pas celui de Top-K")
for prompt in ("le chat", "le chat dort", "le chat dort ."):
    pr = probabilites(scores_suivant(modele, tok.encoder(prompt)))
    gardes = int((filtrer_top_p(pr, 0.9) > 0).sum())
    print(f"  après {prompt!r:<17} : Top-P (0,9) garde {gardes} mot(s) ; le favori a {pr.max().item():.0%}")
print("  Modèle sûr de lui -> peu de mots gardés. Modèle qui hésite -> plus de mots.")

print("\nÉTAPE 4 : 2000 tirages après « le chat » pour chaque réglage")
reglages = [
    ("T = 0,5", probabilites(scores, 0.5)),
    ("T = 1", probabilites(scores, 1.0)),
    ("T = 2", probabilites(scores, 2.0)),
    ("T = 1, Top-K 3", filtrer_top_k(base, 3)),
    ("T = 2, Top-K 3", filtrer_top_k(probabilites(scores, 2.0), 3)),
    ("T = 1, Top-P 0,9", filtrer_top_p(base, 0.9)),
]
g = torch.Generator().manual_seed(0)
print(f"  {'réglage':<18} {'vrai verbe':>10} {'mots distincts':>16}")
for nom, pr in reglages:
    tires = [choisir(pr, g) for _ in range(2000)]
    vrais = sum(t in verbes for t in tires) / len(tires)
    print(f"  {nom:<18} {vrais:>10.1%} {len(set(tires)):>16}")
print("  T = 2 produit des mots absurdes ; Top-K ou Top-P les éliminent tout en gardant de la variété.")

print("\nÉTAPE 5 : la figure")
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
ordre = base.argsort(descending=True)[:7]
for ax, (nom, pr) in zip(axes.flat, reglages):
    ax.bar([mots[int(i)] for i in ordre], [pr[i].item() for i in ordre], color="tab:blue")
    ax.set_title(nom)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
for ax in axes[:, 0]:
    ax.set_ylabel("probabilité")
fig.suptitle("Après « le chat » : probabilités de chaque mot selon le réglage")
plt.tight_layout()
sortie = Path(__file__).resolve().parents[1] / "fiches" / "02_echantillonnage.png"
plt.savefig(sortie, dpi=130)
print(f"  figure : {sortie.relative_to(sortie.parents[3])}")
