"""06_corpus — Leçon 8 : faire écrire le modèle entraîné sur les livres.

    .venv/bin/python src/06_corpus/scripts/08_generation_corpus.py

On recharge livres_meilleur.pt et on génère du texte avec le tokenizer BPE de
4 000 fusions. Rien n'est réappris : les poids ne changent pas. On règle seulement
la façon de CHOISIR le token suivant, avec les fonctions de 05_inference :

  température : divise les scores avant le softmax (basse = prudent, haute = audacieux)
  Top-K       : ne tire qu'entre les K tokens les plus probables
  Top-P       : ne tire qu'entre les tokens qui cumulent la probabilité p

Ici le modèle tourne sur le CPU (5,5 M de paramètres : assez rapide pour quelques
centaines de tokens), ce qui laisse le GPU libre pendant un entraînement.

ÉTAPE 1 : charger le modèle et le tokenizer.
ÉTAPE 2 : un même début, plusieurs réglages (même graine : seul le réglage change).
ÉTAPE 3 : plusieurs débuts avec un réglage raisonnable.

Pour tester d'autres débuts ou réglages, modifier DEBUTS et REGLAGES ci-dessous.
Une seule variable à la fois quand on compare.
"""

import sys
from pathlib import Path

import torch

CHAPITRE = Path(__file__).resolve().parents[1]
SRC = CHAPITRE.parent
sys.path.insert(0, str(CHAPITRE / "tools"))
sys.path.insert(0, str(SRC / "04_training" / "tools"))
sys.path.insert(0, str(SRC / "05_inference" / "tools"))
from bpe_rapide import BPE  # noqa: E402
from checkpoint import DOSSIER_CHECKPOINTS  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE  # noqa: E402
from generation import generer  # noqa: E402
from gpt_torch import GPT  # noqa: E402

# --- réglages ---
N_FUSIONS = 4000
CHECKPOINT = "livres_meilleur.pt"
N_TOKENS = 120
GRAINE = 0
DEBUT = "Il était une fois"
DEBUTS = ["Il était une fois", "- Qu'est-ce que tu fais ? demanda-t-il.\n-", "La nuit tombait sur la ville"]
REGLAGES = [  # (température, top_k, top_p)
    (0.3, None, None),
    (0.8, None, None),
    (1.2, None, None),
    (0.8, 40, None),
    (0.8, None, 0.9),
]

print("ÉTAPE 1 — charger le modèle")
bpe = BPE.charger(DOSSIER_TOKENISE / f"bpe_{N_FUSIONS}.json")
contenu = torch.load(DOSSIER_CHECKPOINTS / CHECKPOINT, map_location="cpu")
modele = GPT(**contenu["config"])
modele.load_state_dict(contenu["poids"])
modele.eval()
print(f"  {CHECKPOINT} : étape {contenu['etape']}, perte de validation {contenu['perte_val']:.3f}")


def ecrire(debut, temperature, top_k=None, top_p=None):
    generateur = torch.Generator().manual_seed(GRAINE)
    ids = generer(modele, bpe.encoder(debut), N_TOKENS, temperature, top_k, top_p, generateur)
    return bpe.decoder(ids)


print("\nÉTAPE 2 — un début, plusieurs réglages")
print(f"  début : {DEBUT!r}, graine {GRAINE}, {N_TOKENS} tokens")
for temperature, top_k, top_p in REGLAGES:
    print(f"\n  température {temperature}, top_k {top_k}, top_p {top_p}")
    print("  " + ecrire(DEBUT, temperature, top_k, top_p).replace("\n", "\n  "))

print("\nÉTAPE 3 — plusieurs débuts (température 0.8, top_p 0.9)")
for debut in DEBUTS:
    print(f"\n  début : {debut!r}")
    print("  " + ecrire(debut, 0.8, None, 0.9).replace("\n", "\n  "))
