"""06_corpus — Leçon 13 : ce que le modèle a mis dans ses embeddings.

Rappel (leçon 7) : le modèle commence par transformer chaque token en un
vecteur de d nombres : son « embedding » (ligne de la table `emb_mots`, de
taille 4 612 × d). Au départ ces nombres sont tirés au hasard. Personne ne
dit au modèle ce que veut dire « chat » : si deux tokens ont des vecteurs
proches à la fin, c'est que l'entraînement les a rapprochés, parce qu'ils
servent dans les mêmes contextes.

Cette leçon regarde ce qui a été appris :
  1. la similarité cosinus : à quel point deux vecteurs pointent dans la même
     direction. cos(a, b) = (a · b) / (|a| |b|). Elle vaut 1 si les vecteurs
     ont la même direction, 0 s'ils sont indépendants, -1 s'ils sont opposés ;
  2. les « voisins » d'un token : les tokens de plus grand cosinus ;
  3. l'analogie par addition de vecteurs (essai, parfois raté) ;
  4. l'ACP (analyse en composantes principales), écrite à la main : on
     cherche les 2 directions où les vecteurs varient le plus, pour dessiner
     d dimensions sur une feuille. Elle perd de l'information : une carte
     n'est pas le territoire.

Tout tourne sur le CPU, en quelques secondes. Figure : logs/embeddings_acp.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CHAPITRE = Path(__file__).resolve().parents[1]
RACINE = CHAPITRE.parents[1]
SRC = CHAPITRE.parent
sys.path.insert(0, str(CHAPITRE / "tools"))
sys.path.insert(0, str(SRC / "04_training" / "tools"))
from bpe_rapide import BPE  # noqa: E402
from checkpoint import DOSSIER_CHECKPOINTS  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE, charger_bin  # noqa: E402
from gpt_torch import GPT  # noqa: E402

CHECKPOINT = "livres_T256_meilleur.pt"
MOTS = [" chat", " chien", " homme", " femme", " roi", " reine", " père", " mère",
        " nuit", " jour", " triste", " heureux", " rouge", " bleu", "1", "2"]
N_VOISINS = 6
N_POINTS = 150  # tokens les plus fréquents à dessiner

bpe = BPE.charger(DOSSIER_TOKENISE / "bpe_4000.json")
contenu = torch.load(DOSSIER_CHECKPOINTS / CHECKPOINT, map_location="cpu")
modele = GPT(**contenu["config"])
modele.load_state_dict(contenu["poids"])
E = modele.emb_mots.weight.detach().numpy()  # une ligne par token
unitaires = E / np.linalg.norm(E, axis=1, keepdims=True)  # |v| = 1 : cos = produit scalaire


def montrer(i):
    return repr(bpe.decoder([i]))


def identifiant(mot):
    """L'id du mot s'il forme UN seul token, sinon None."""
    ids = bpe.encoder(mot)
    return ids[0] if len(ids) == 1 else None


print(f"ÉTAPE 1 — la table : {E.shape[0]} tokens × {E.shape[1]} nombres ({CHECKPOINT})")
hasard = np.random.default_rng(0).integers(0, len(E), size=(2000, 2))
cos_hasard = (unitaires[hasard[:, 0]] * unitaires[hasard[:, 1]]).sum(axis=1)
print(f"  cosinus moyen entre deux tokens au hasard : {cos_hasard.mean():+.3f} (écart {cos_hasard.std():.3f})")

print(f"\nÉTAPE 2 — les {N_VOISINS} plus proches voisins (cosinus)")
for mot in MOTS:
    i = identifiant(mot)
    if i is None:
        print(f"  {mot!r} : découpé en plusieurs tokens, ignoré")
        continue
    cos = unitaires @ unitaires[i]
    voisins = [j for j in np.argsort(-cos) if j != i][:N_VOISINS]
    print(f"  {mot!r:<10} -> " + ", ".join(f"{montrer(j)} {cos[j]:.2f}" for j in voisins))

print("\nÉTAPE 3 — analogie : roi - homme + femme = ?")
ids = [identifiant(m) for m in (" roi", " homme", " femme")]
if None in ids:
    print("  un des mots n'est pas un token unique")
else:
    cible = E[ids[0]] - E[ids[1]] + E[ids[2]]
    cos = unitaires @ (cible / np.linalg.norm(cible))
    meilleurs = [j for j in np.argsort(-cos) if j not in ids][:5]
    print("  plus proches :", ", ".join(f"{montrer(j)} {cos[j]:.2f}" for j in meilleurs))
    print("  (un petit modèle de 5 millions de paramètres réussit rarement : c'est instructif)")

print(f"\nÉTAPE 4 — ACP à la main sur les {N_POINTS} tokens les plus fréquents")
frequences = np.bincount(charger_bin("validation"), minlength=len(E))
frequents = np.argsort(-frequences)[:N_POINTS]
X = E[frequents] - E[frequents].mean(axis=0)  # on centre le nuage
valeurs, vecteurs = np.linalg.eigh(np.cov(X.T))  # directions de variation, de la plus faible à la plus forte
axes = vecteurs[:, ::-1][:, :2]
part = valeurs[::-1][:2].sum() / valeurs.sum()
print(f"  les 2 axes retenus gardent {part:.0%} de la variation totale")
points = X @ axes

fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(points[:, 0], points[:, 1], s=8, color="tab:blue")
for (x, y), i in zip(points, frequents):
    ax.annotate(bpe.decoder([int(i)]).replace("\n", "↵"), (x, y), fontsize=8)
ax.set_title(f"Embeddings des {N_POINTS} tokens les plus fréquents (ACP, {part:.0%} de la variation)")
ax.set_xlabel("axe 1")
ax.set_ylabel("axe 2")
fig.tight_layout()
sortie = RACINE / "logs" / "embeddings_acp.png"
fig.savefig(sortie, dpi=110)
print(f"  figure : {sortie.relative_to(RACINE)}")
