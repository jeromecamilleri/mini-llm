"""
LEÇON 3 (inference) : comparer des modèles sur les mêmes prompts
================================================================

    .venv/bin/python src/05_inference/scripts/03_comparaison.py

PLACE DANS LE PARCOURS (chapitre 05_inference)
Leçons 1 et 2 : on génère et on règle le tirage avec UN modèle. Ici on compare
TROIS modèles qui ne diffèrent que d'UNE chose, le moment de l'entraînement
(règle du projet : changer une seule variable à la fois) :

  - AU HASARD  : le modèle de départ, jamais entraîné ;
  - MEILLEUR   : checkpoint de l'étape 80 (validation minimale, leçon 3 de 04_training) ;
  - DERNIER    : checkpoint de l'étape 600 (a appris par cœur).

Même prompt, même réglage du tirage (T = 1), même graine : seul le modèle change.

LA QUESTION : la perte de validation prédit-elle la qualité du texte ?
Réponse de l'expérience : pas ici ! Le dernier modèle a la pire validation des deux
entraînés mais écrit le texte le plus correct (voir ÉTAPE 3). Mesurer, c'est aussi
découvrir que sa mesure ne dit pas ce qu'on croyait.
Deux mesures :
  1. la perte de validation (04_training) : « à quel point le modèle est surpris
     par du texte jamais vu » ;
  2. la part de PHRASES CORRECTES parmi les phrases générées. Une phrase est
     correcte si elle suit la grammaire du corpus :
         (le chat | le chien | le cheval | l ' oiseau) (dort | mange | court) [aussi]
     Cette règle ne sert qu'à NOTER le texte ; le modèle ne la connaît pas.

Avec un vrai corpus on ne peut pas écrire une grammaire : on lira le texte, on
mesurera la perplexité, ou on utilisera un autre modèle comme juge. Ici le corpus
est assez petit pour que la grammaire existe.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generation import charger_modele, generer  # (ajoute aussi tools de 04_training au chemin)
from checkpoint import DOSSIER_CHECKPOINTS, charger
from donnees import charger_corpus, tirer_batch
from gpt_torch import GPT, perte

tok, _ = charger_modele()
_, _, validation = charger_corpus()
meilleur, infos_meilleur = charger(DOSSIER_CHECKPOINTS / "animaux_meilleur.pt", GPT)
dernier, infos_dernier = charger(DOSSIER_CHECKPOINTS / "animaux_dernier.pt", GPT)
torch.manual_seed(0)
hasard = GPT(**infos_meilleur["config"])  # mêmes dimensions, poids tirés au hasard
modeles = [("au hasard", hasard), ("meilleur (étape 80)", meilleur), ("dernier (étape 600)", dernier)]
for _, m in modeles:
    m.eval()

GRAMMAIRE = re.compile(r"(le (chat|chien|cheval)|l ' oiseau) (dort|mange|court)( aussi)?")


def perte_validation(m, T=4):
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        return float(np.mean([perte(m, *tirer_batch(validation, 16, T, gen)).item() for _ in range(20)]))


def phrases_completes(ids):
    """Découpe le texte généré en phrases terminées par « . » (la dernière, coupée, est ignorée)."""
    morceaux = tok.decoder(ids).split(" . ")
    return morceaux[:-1] if len(morceaux) > 1 else []


print("ÉTAPE 1 : mêmes prompts, trois modèles (tirage, T = 1, même graine)")
for prompt in ("le chat", "l ' oiseau"):
    print(f"  prompt « {prompt} »")
    for nom, m in modeles:
        g = torch.Generator().manual_seed(0)
        texte = tok.decoder(generer(m, tok.encoder(prompt), 12, generateur=g))
        print(f"    {nom:<22} {texte}")

print("\nÉTAPE 2 : mesure sur 200 générations de 24 mots chacune (départ « le »)")
print(f"  {'modèle':<22} {'val.':>6} {'phrases correctes':>20}")
resultats = []
for nom, m in modeles:
    g = torch.Generator().manual_seed(1)
    total = bonnes = 0
    for _ in range(200):
        for phrase in phrases_completes(generer(m, tok.encoder("le"), 24, generateur=g)):
            total += 1
            bonnes += bool(GRAMMAIRE.fullmatch(phrase))
    part = bonnes / total
    resultats.append((nom, perte_validation(m), part))
    print(f"  {nom:<22} {perte_validation(m):>6.3f} {part:>12.1%} ({bonnes}/{total})")

print("\nÉTAPE 3 : que conclure ?")
print("  - AU HASARD : validation proche de ln(13) = 2,56 et 0 % de phrases correctes : il n'a rien appris.")
print("  - DERNIER : la PIRE validation des deux entraînés, mais le MEILLEUR texte (la grammaire est dans")
print("    le corpus d'entraînement, qu'il a mémorisé : il le recrache avec peu de fautes).")
print("  - MEILLEUR : meilleure validation, mais davantage de fautes (tirage de mots peu probables).")
print("  Ce n'est pas un bug : les deux mesures ne répondent pas à la même question.")
print("  La validation (7 mots seulement ici) mesure la surprise sur du texte NEUF ; la grammaire notée")
print("  ici est celle du corpus d'ENTRAÎNEMENT. Avec un vrai corpus, validation et qualité du texte")
print("  évolueraient bien plus ensemble ; avec 67 mots, la validation est trop bruitée pour trancher.")
