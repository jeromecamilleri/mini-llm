"""
LEÇON 3 (training) : checkpoints, meilleur modèle, métriques
============================================================

    .venv/bin/python src/04_training/scripts/03_checkpoints.py

PLACE DANS LE PARCOURS (chapitre 04_training)
Leçon 2 : la validation était la meilleure à l'étape 100, puis elle REMONTAIT
(surapprentissage). Mais à la fin des 600 étapes, on ne gardait que le modèle
final : le moins bon ! Et si le script s'arrête, tout est perdu.
(Image : réviser un examen. Au début, plus on révise, mieux on réussit les exercices
connus ET les questions jamais vues. Ensuite on apprend PAR CŒUR : les exercices
connus vont de mieux en mieux, mais les nouvelles questions sont ratées. La perte de
validation remonte. Le modèle de l'étape 100 valait mieux que celui de l'étape 600.)

L'ESSENTIEL
  1. CHECKPOINT : un fichier qui contient les poids (+ la config pour reconstruire
     le modèle + l'étape). Sans lui, un modèle entraîné disparaît à la fin du script.
  2. MEILLEUR MODÈLE : on mesure la validation souvent ; chaque fois qu'elle bat
     son record, on SAUVE. À la fin, on garde le meilleur, pas le dernier (c'est
     l'ARRÊT ANTICIPÉ de 01_fundamentals, leçon 11, mais appliqué au GPT).
  3. MÉTRIQUES : on écrit les pertes dans un fichier (logs/) pour tracer la courbe
     après coup, sans relancer l'entraînement.

Un checkpoint contient :
  poids       state_dict() : tous les paramètres (le « cerveau »)
  config      taille du vocabulaire, d, têtes, blocs, T_max : de quoi reconstruire la forme
  etape       où l'on en était
  perte_val   la validation à ce moment
  optimiseur  (optionnel) l'état d'AdamW, pour REPRENDRE l'entraînement où il s'était arrêté

Fichiers écrits : models/checkpoints/animaux_meilleur.pt, animaux_dernier.pt,
logs/entrainement_animaux.csv, et les figures fiches/03_checkpoints.png et
03_paysage.png (ÉTAPE 6 : un jouet à UN poids pour distinguer minimum LOCAL, trouvé
par la descente, et MEILLEUR CHECKPOINT, choisi par la validation).
"""
import csv
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")  # pas de fenêtre : on enregistre l'image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from checkpoint import DOSSIER_CHECKPOINTS, DOSSIER_LOGS, charger, sauver
from donnees import charger_corpus, tirer_batch
from gpt_torch import GPT, perte

torch.manual_seed(0)
tok, entrainement, validation = charger_corpus()
config = dict(taille_vocab=len(tok.vocab), d=32, n_tetes=2, n_blocs=2, T_max=4)
T = config["T_max"]
modele = GPT(**config)
optimiseur = torch.optim.AdamW(modele.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(0)
chemin_meilleur = DOSSIER_CHECKPOINTS / "animaux_meilleur.pt"
chemin_dernier = DOSSIER_CHECKPOINTS / "animaux_dernier.pt"


def moyenne_perte(m, donnees, n_batchs=20):
    gen = torch.Generator().manual_seed(123)  # mêmes fenêtres à chaque mesure
    m.eval()
    with torch.no_grad():
        total = sum(perte(m, *tirer_batch(donnees, 16, T, gen)).item() for _ in range(n_batchs))
    m.train()
    return total / n_batchs


def phrase(m, debut, n=8):
    ids = torch.tensor([tok.encoder(debut)])
    return tok.decoder(m.generer(ids, n)[0].tolist())


print("ÉTAPE 1 : entraînement, avec mesure toutes les 20 étapes")
historique = []  # (étape, perte train, perte validation)
meilleure = float("inf")
for etape in range(1, 601):
    x, y = tirer_batch(entrainement, 16, T, g)
    l = perte(modele, x, y)
    optimiseur.zero_grad()
    l.backward()
    optimiseur.step()
    if etape % 20 == 0:
        pt, pv = moyenne_perte(modele, entrainement), moyenne_perte(modele, validation)
        historique.append((etape, pt, pv))
        if pv < meilleure:  # record battu : on sauve CE modèle
            meilleure = pv
            sauver(chemin_meilleur, modele, config, etape, pv, optimiseur)
            print(f"  étape {etape:>3} : train {pt:.3f}  val {pv:.3f}  <- nouveau record, checkpoint sauvé")
sauver(chemin_dernier, modele, config, 600, historique[-1][2], optimiseur)
print(f"  fin : dernier modèle sauvé aussi (val {historique[-1][2]:.3f}), meilleur val {meilleure:.3f}")

print("\nÉTAPE 2 : les métriques dans un fichier")
DOSSIER_LOGS.mkdir(exist_ok=True)
with open(DOSSIER_LOGS / "entrainement_animaux.csv", "w", newline="") as f:
    ecrivain = csv.writer(f)
    ecrivain.writerow(["etape", "perte_train", "perte_val"])
    ecrivain.writerows(historique)
print(f"  logs/entrainement_animaux.csv : {len(historique)} lignes (étape, perte train, perte val)")

print("\nÉTAPE 3 : recharger le meilleur modèle depuis le fichier")
meilleur, contenu = charger(chemin_meilleur, GPT)
print(f"  checkpoint de l'étape {contenu['etape']}, val enregistrée {contenu['perte_val']:.3f}")
print(f"  val recalculée avec le modèle rechargé : {moyenne_perte(meilleur, validation):.3f}  (identique)")
print("  Les poids sont revenus tels quels : le fichier suffit, plus besoin de ré-entraîner.")

print("\nÉTAPE 4 : meilleur modèle contre dernier modèle")
for nom, m in (("meilleur", meilleur), ("dernier ", modele)):
    print(f"  {nom} : train {moyenne_perte(m, entrainement):.3f}  val {moyenne_perte(m, validation):.3f}"
          f"  | {phrase(m, 'le chat')}")
print("  Le meilleur est moins bon sur l'entraînement mais meilleur sur ce qu'il n'a jamais vu.")

print("\nÉTAPE 5 : la courbe")
etapes, pts, pvs = zip(*historique)
meilleur_etape = contenu["etape"]
plt.figure(figsize=(10, 6))
plt.plot(etapes, pts, label="entraînement")
plt.plot(etapes, pvs, label="validation")
plt.axvline(meilleur_etape, color="gray", linestyle="--", label=f"meilleur modèle (étape {meilleur_etape})")
plt.xlabel("étape")
plt.ylabel("perte")
plt.title("Surapprentissage : la validation remonte, l'entraînement continue de baisser")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
sortie = Path(__file__).resolve().parents[1] / "fiches" / "03_checkpoints.png"
plt.savefig(sortie, dpi=130)
print(f"  courbe : {sortie.relative_to(sortie.parents[3])}")

# ---------------------------------------------------------------
# ÉTAPE 6 : le « paysage » (jouet à UN seul poids w, pour voir la différence)
# ---------------------------------------------------------------
print("\nÉTAPE 6 : le paysage de la perte (jouet à un seul poids)")
import numpy as np


def perte_train(w):  # un creux profond à gauche (global), un creux moyen à droite (local)
    return 0.02 * w ** 4 - 0.3 * w ** 2 + 0.3 * w + 2


def pente(w):
    return 0.08 * w ** 3 - 0.6 * w + 0.3


def perte_val(w):  # même forme, mais pénalisée quand on s'éloigne trop : « par cœur »
    return perte_train(w) + 0.12 * (w - 0.8) ** 2 * (1 + 0.3 * abs(w - 0.8))


w = 0.8
chemin = [w]
for _ in range(60):
    w -= 0.15 * pente(w)  # descente de gradient
    chemin.append(w)
chemin = np.array(chemin)
meilleur_pas = int(np.argmin(perte_val(chemin)))
print(f"  la descente part de w = 0,8 et s'arrête en w = {chemin[-1]:.2f} (perte train {perte_train(chemin[-1]):.2f})")
print("  C'est un minimum LOCAL : il existe un creux bien plus profond à gauche (w = -2,7).")
print(f"  Mais la validation était la meilleure au pas {meilleur_pas} (w = {chemin[meilleur_pas]:.2f}), pas à la fin.")

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6))
ws = np.linspace(-4, 4, 400)
a.plot(ws, perte_train(ws), color="tab:blue", label="perte d'entraînement")
a.plot(ws, perte_val(ws), color="tab:orange", label="perte de validation")
a.plot(chemin, perte_train(chemin), "o-", color="black", markersize=3, label="chemin de la descente")
a.plot(chemin[0], perte_train(chemin[0]), "s", color="green", markersize=10, label="départ")
a.plot(chemin[-1], perte_train(chemin[-1]), "v", color="red", markersize=11, label="arrivée : minimum local")
a.plot(-2.7, perte_train(-2.7), "*", color="purple", markersize=15, label="minimum global (jamais trouvé)")
a.plot(chemin[meilleur_pas], perte_val(chemin[meilleur_pas]), "D", color="tab:orange",
       markeredgecolor="black", markersize=10, label="meilleur point de validation")
a.set_xlabel("valeur du poids w")
a.set_ylabel("perte")
a.set_ylim(-0.3, 4)
a.set_title("1. Le paysage : la descente tombe dans le premier creux")
a.legend(fontsize=9)
a.grid(alpha=0.3)

pas = np.arange(len(chemin))
b.plot(pas, perte_train(chemin), color="tab:blue", label="entraînement")
b.plot(pas, perte_val(chemin), color="tab:orange", label="validation")
b.axvline(meilleur_pas, color="gray", linestyle="--", label=f"meilleur checkpoint (pas {meilleur_pas})")
b.set_xlabel("pas de la descente")
b.set_ylabel("perte")
b.set_title("2. Le même trajet dans le temps : on garde la validation minimale")
b.legend(fontsize=9)
b.grid(alpha=0.3)
plt.tight_layout()
sortie2 = Path(__file__).resolve().parents[1] / "fiches" / "03_paysage.png"
plt.savefig(sortie2, dpi=130)
print(f"  figure : {sortie2.relative_to(sortie2.parents[3])}")
