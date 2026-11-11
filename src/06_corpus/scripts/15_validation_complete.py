"""06_corpus — Leçon 15 : mesurer la perte sur TOUTE la validation, sans perdre de contexte.

    .venv/bin/python src/06_corpus/scripts/15_validation_complete.py                 # CPU, 3 modèles
    .venv/bin/python src/06_corpus/scripts/15_validation_complete.py cuda            # même chose sur le GPU
    .venv/bin/python src/06_corpus/scripts/15_validation_complete.py --test          # essai rapide (50 fenêtres)
    .venv/bin/python src/06_corpus/scripts/15_validation_complete.py livres_P4_8blocs_meilleur.pt cuda
    .venv/bin/python src/06_corpus/scripts/15_validation_complete.py --validation=theatre_validation   # dialogues (leçon 17)

Arguments (dans n'importe quel ordre) : `cpu` (défaut) ou `cuda`, `--test`, des noms de
fichiers de models/checkpoints/ (sinon la liste MODELES ci-dessous), et
`--validation=NOM` : un fichier de datasets/tokenized/ (défaut : `validation`, les 17 livres).
Avec `theatre_validation`, on mesure les modèles sur des dialogues qu'ils n'ont jamais lus :
c'est la référence à avoir AVANT d'ajouter du théâtre à l'apprentissage (leçon 17).

Pourquoi cette leçon. La perte de validation affichée pendant l'entraînement (leçon 7)
est une mesure rapide, avec quatre faiblesses :
  1. elle ne lit que 40 batchs de 32 fenêtres, soit environ 11 % de la validation ;
  2. ces fenêtres sont tirées au hasard : elles peuvent se chevaucher, et des zones ne sont
     jamais lues ;
  3. au début d'une fenêtre, le modèle prédit presque sans contexte (le 2e token avec 1 seul
     token lu, le 256e avec 255) : ces premiers tokens sont mal prédits, ce qui gonfle la perte
     par rapport à une lecture continue d'un livre ;
  4. avec T = 128 (référence) et T = 256 (les autres), le contexte n'est pas le même.

Ce que fait cette leçon.
  - Elle lit TOUTE la validation (3 046 367 tokens), du début à la fin.
  - Fenêtre glissante : chaque fenêtre de T tokens avance de PAS = T/2 tokens ; on ne compte
    que les T/2 derniers tokens de chaque fenêtre (la première est comptée en entier). Chaque
    token de la validation est donc compté UNE seule fois, avec au moins T/2 tokens de contexte.
  - Elle affiche la perte, la perplexité et les BITS PAR OCTET : cette dernière unité ne
    dépend ni du tokenizer ni de T, et se compare à un autre modèle (voir fiche 15).
  - Option AUSSI_SANS_CONTEXTE : refait la mesure avec des fenêtres disjointes qui comptent
    tous leurs tokens (début de fenêtre compris) ; la différence est l'effet de la faiblesse 3.

Attention : D n'est évalué ici que sur la validation non filtrée (celle de l'entraînement).
Pour la comparaison A / D sur validation filtrée, voir la leçon 14.

Coût : sur 4 fils de CPU, environ 10 minutes pour B et 15 à 20 minutes pour A, D ou P4
(extrapolé d'un essai sur 50 fenêtres ; T/2 de recouvrement, donc 2 fois la lecture) ; sur le
GTX 1070 nettement moins (non mesuré). Si un entraînement tourne sur le GPU, rester sur le CPU :
les deux se ralentiraient.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

CHAPITRE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPITRE / "tools"))
sys.path.insert(0, str(CHAPITRE.parent / "04_training" / "tools"))
from checkpoint import DOSSIER_CHECKPOINTS  # noqa: E402
from bpe_rapide import BPE  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE, charger_bin  # noqa: E402
from gpt_torch import GPT  # noqa: E402

# --- réglages ---
MODELES = ["livres_T256_meilleur.pt", "livres_A_d384_meilleur.pt", "livres_D_filtre_meilleur.pt"]
TAILLE_BATCH = 32
OCTETS_VALIDATION = 9_978_076  # octets UTF-8 des 17 livres de validation (mesuré lors de la comparaison en bits par octet)
AUSSI_SANS_CONTEXTE = False  # True : ajoute la mesure à fenêtres disjointes (début de fenêtre compté), pour voir le biais
N_FILS_CPU = 4

args = sys.argv[1:]
appareil = "cuda" if "cuda" in args else "cpu"
essai = "--test" in args
fichiers = [a for a in args if a.endswith(".pt")] or MODELES
nom_validation = next((a.split("=", 1)[1] for a in args if a.startswith("--validation=")), "validation")
if appareil == "cuda" and not torch.cuda.is_available():
    sys.exit("cuda demandé mais indisponible : relancer sans l'argument cuda.")
torch.set_num_threads(N_FILS_CPU)

validation = charger_bin(nom_validation)
if nom_validation == "validation":
    octets = OCTETS_VALIDATION
else:
    # autre validation : octets du texte décodé (approximatif si des tokens inconnus <inc> sont perdus)
    octets = len(BPE.charger(DOSSIER_TOKENISE / "bpe_4000.json").decoder(validation.tolist()).encode("utf-8"))
tokens_par_octet = len(validation) / octets
print("ÉTAPE 1 — la validation")
print(f"  {nom_validation} : {len(validation):,} tokens, {octets:,} octets ({tokens_par_octet:.3f} token par octet)")
print(f"  appareil : {appareil}" + (" — mode --test : 50 fenêtres seulement" if essai else ""))


def perte_complete(modele, donnees, T, pas):
    """Perte moyenne par token, chaque token de la validation étant compté une seule fois.

    Fenêtres de T tokens qui avancent de `pas`. Le token prédit à la position j de la fenêtre
    qui commence en s est donnees[s + j + 1]. La première fenêtre compte toutes ses positions ;
    les suivantes ne comptent que leurs `pas` dernières (les autres ont déjà été comptées).
    Avec pas = T, les fenêtres sont disjointes et tout compte, début de fenêtre compris.
    """
    debuts = np.arange(0, len(donnees) - T - 1, pas)
    if essai:
        debuts = debuts[:50]
    somme, compte = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(debuts), TAILLE_BATCH):
            lot = debuts[i : i + TAILLE_BATCH]
            x = torch.from_numpy(np.stack([donnees[s : s + T] for s in lot]).astype(np.int64)).to(appareil)
            y = torch.from_numpy(np.stack([donnees[s + 1 : s + T + 1] for s in lot]).astype(np.int64)).to(appareil)
            erreurs = torch.nn.functional.cross_entropy(modele(x).transpose(1, 2), y, reduction="none")  # (B, T) : l'erreur de chaque position
            for k, s in enumerate(lot):
                premiere = (s == 0)
                depuis = 0 if premiere else T - pas
                somme += erreurs[k, depuis:].sum().item()
                compte += T - depuis
    return somme / compte, compte


print("\nÉTAPE 2 — perte de chaque modèle sur la validation (chaque token compté une fois)")
resultats = []
for fichier in fichiers:
    contenu = torch.load(DOSSIER_CHECKPOINTS / fichier, map_location=appareil)
    modele = GPT(**contenu["config"]).to(appareil)
    modele.load_state_dict(contenu["poids"])
    modele.eval()
    T = contenu["config"]["T_max"]
    debut = time.time()
    p, n = perte_complete(modele, validation, T, pas=T // 2)
    bpo = p * tokens_par_octet / math.log(2)
    ligne = dict(fichier=fichier, T=T, etape=contenu.get("etape", "?"), perte=p, bpo=bpo, n=n, rapide=contenu.get("perte_val") if nom_validation == "validation" else None)
    print(f"  {fichier:30s} T {T:3d} (étape {ligne['etape']}) : perte {p:.4f} (perplexité {math.exp(p):.1f}), "
          f"{bpo:.3f} bit/octet  [{n:,} tokens, {time.time() - debut:.0f} s]")
    if AUSSI_SANS_CONTEXTE:
        p0, n0 = perte_complete(modele, validation, T, pas=T)
        ligne["sans_contexte"] = p0
        print(f"  {'':30s} fenêtres disjointes (début compté) : perte {p0:.4f}  -> biais du début de fenêtre : {p0 - p:+.4f}")
    resultats.append(ligne)

print("\nÉTAPE 3 — bilan")
print(f"  {'modèle':30s} {'mesure rapide':>14s} {'complète':>9s} {'écart':>7s} {'bit/octet':>10s}")
for r in resultats:
    rapide = f"{r['rapide']:.4f}" if r["rapide"] is not None else "?"
    ecart = f"{r['perte'] - r['rapide']:+.4f}" if r["rapide"] is not None else "?"
    print(f"  {r['fichier']:30s} {rapide:>14s} {r['perte']:9.4f} {ecart:>7s} {r['bpo']:10.3f}")
print("  mesure rapide = celle de l'entraînement (40 batchs au hasard) ; complète = toute la validation, avec contexte.")
print("  Un écart négatif est attendu : la fenêtre glissante donne du contexte à chaque token.")
if essai:
    print("  ATTENTION : en mode --test, seul le tout début de la validation est lu : ces chiffres ne sont pas représentatifs.")
