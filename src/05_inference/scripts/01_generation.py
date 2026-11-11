"""
LEÇON 1 (inference) : générer du texte avec le modèle entraîné
==============================================================

    .venv/bin/python src/05_inference/scripts/01_generation.py

PLACE DANS LE PARCOURS (chapitre 05_inference)
Chapitre 04_training : on a entraîné le GPT et sauvegardé le meilleur modèle
(models/checkpoints/animaux_meilleur.pt). Ici on le RECHARGE depuis le fichier et
on s'en sert : c'est l'INFÉRENCE (utiliser un modèle déjà entraîné, sans le modifier).

L'ESSENTIEL : la génération AUTORÉGRESSIVE (« qui se nourrit de sa propre sortie »)
  1. on encode le début (le prompt) en identifiants ;
  2. le modèle donne un score pour chaque mot ; on regarde la DERNIÈRE position ;
  3. softmax : scores -> probabilités ;
  4. on CHOISIT un mot (glouton ou tirage au sort) ;
  5. on l'ajoute au texte, et on recommence à l'étape 2.
Le modèle ne prédit jamais « la phrase » : seulement UN mot, à chaque tour.

À CHAQUE TOUR, le modèle relit tout le texte (dans la limite de sa fenêtre T_max,
ici 4 mots) : il N'A PAS de mémoire entre deux tours. Ce qu'il « se rappelle »,
c'est seulement ce qui est encore dans la fenêtre.

Pas de gradient ici (torch.no_grad) : les poids ne bougent pas.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generation import charger_modele, generer, probabilites, scores_suivant

tok, modele = charger_modele()
mots = tok.vocab.id_vers_token

print("ÉTAPE 1 : le modèle rechargé depuis le fichier")
print(f"  {modele.nb_parametres()} paramètres, fenêtre T_max = {modele.T_max} mots")

print("\nÉTAPE 2 : un seul tour, au ralenti (prompt « le chat »)")
ids = tok.encoder("le chat")
print(f"  identifiants : {ids}")
probas = probabilites(scores_suivant(modele, ids))
for i in probas.argsort(descending=True)[:5]:
    print(f"  {mots[int(i)]:<8} {probas[i].item():6.1%}")
print("  Après « le chat » : dort, mange ou court (les trois verbes du corpus après chat).")

print("\nÉTAPE 3 : la boucle autorégressive, mot par mot (glouton)")
ids = tok.encoder("le chat")
for _ in range(7):
    probas = probabilites(scores_suivant(modele, ids))
    suivant = int(probas.argmax())
    vue = tok.decoder(ids[-modele.T_max:])
    print(f"  voit [{vue:<18}] -> {mots[suivant]:<6} ({probas[suivant].item():.0%})")
    ids.append(suivant)
print(f"  texte final : {tok.decoder(ids)}")

print("\nÉTAPE 4 : le modèle oublie ce qui sort de la fenêtre")
print("  La fenêtre fait 4 mots : au 4e tour le modèle voit « chat dort . le » et le premier « le » est sorti.")
print("  Il choisit `cheval` sans savoir que le texte parlait d'un chat : il n'a pas de mémoire plus longue.")
print("  Ici ce n'est pas grave : le corpus est très régulier.")

print("\nÉTAPE 5 : glouton contre tirage au sort (voir leçon 7 de 03_transformer)")
print("  glouton :", tok.decoder(generer(modele, tok.encoder("le chat"), 10, temperature=1.0)))
g = torch.Generator().manual_seed(0)
for essai in range(3):
    print(f"  tirage {essai + 1} :", tok.decoder(generer(modele, tok.encoder("le chat"), 10, generateur=g)))
print("  Glouton : toujours la même phrase. Tirage : une phrase différente à chaque essai,")
print("  parfois fautive (« le chat chien mange » : `chien` avait une petite probabilité et a été tiré).")

print("\nÉTAPE 6 : un mot inconnu")
inconnu = tok.encoder("le tigre dort")
print(f"  « le tigre dort » -> {inconnu} : « tigre » devient <inc> (identifiant 0), absent du corpus.")
print("  Suite :", tok.decoder(generer(modele, inconnu, 6)))
print("  Le modèle ne connaît que 13 mots : hors vocabulaire, il ne peut ni comprendre ni produire.")
