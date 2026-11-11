"""06_corpus — Leçon 7 : entraîner un GPT sur les livres.

    .venv/bin/python src/06_corpus/scripts/07_entrainement_corpus.py

C'est la même boucle que 04_training (leçons 2 et 3) : tirer un batch, calculer
la perte, rétropropager, faire un pas d'AdamW, mesurer la validation, sauver le
meilleur modèle. Ce qui change, c'est l'échelle :

                      04_training (animaux)     ici (livres)
  corpus              67 mots                   66 millions de tokens
  vocabulaire         13 mots                   4 612 tokens (BPE)
  modèle              26 000 paramètres         environ 5,5 millions
  contexte T          4                         128
  GPU                 inutile                   indispensable

TROIS AJOUTS pour que l'entraînement reste stable sur un modèle plus gros :
  1. ÉCHAUFFEMENT (warmup) : le taux d'apprentissage part de 0 et monte jusqu'à
     LR_MAX en WARMUP étapes. Au tout début, les poids sont aléatoires et un
     grand pas ferait n'importe quoi.
  2. DÉCROISSANCE (cosine) : ensuite le taux redescend doucement vers LR_MIN. Les
     grands pas servent à dégrossir, les petits à affiner. Image : pour poser un
     objet à un endroit précis, on approche vite de la main, puis on ralentit au
     fur et à mesure qu'on s'en rapproche. La courbe suit la forme d'un cosinus
     (de 0 à π) : elle part à plat, descend vite au milieu, puis finit à plat.
     Avec LR_MAX = 1e-3, LR_MIN = 1e-4 et 20 000 étapes :
         étape 300   : 1,0e-3     étape 10 000 : 5,6e-4
         étape 5 000 : 8,8e-4     étape 15 000 : 2,4e-4     étape 20 000 : 1,0e-4
     C'est pourquoi la perte ralentit vers la fin : les pas sont plus petits.
  3. ÉCRÊTAGE DES GRADIENTS (clip) : si la norme du gradient dépasse 1, on le
     ramène à 1. Un seul batch bizarre ne peut plus casser le modèle.

REPÈRES POUR LIRE LA PERTE
  - Un modèle au hasard prédit chaque token avec la même probabilité :
    perte = ln(4612) = 8,44.
  - La perte est la « surprise » moyenne en nats. exp(perte) = perplexité : le
    nombre de tokens parmi lesquels le modèle hésite. Perte 5 -> hésite entre 150
    tokens ; perte 4 -> entre 55 ; perte 3 -> entre 20.
  - Ne descend jamais à 0 : après « il » beaucoup de mots sont possibles.

TAILLE_BATCH : nombre de fenêtres de T tokens que le modèle lit AVANT de corriger ses
poids une fois. À chaque étape on tire TAILLE_BATCH extraits de livres, on calcule la
perte moyenne sur eux, puis un seul pas de correction. Tokens lus par étape =
TAILLE_BATCH × T (ici 32 × 128 = 4 096). Un batch grand : correction plus fiable mais
plus de mémoire GPU par étape ; un batch petit : correction plus bruitée, moins de
mémoire. Quand T double, on divise le batch par 2 pour garder les mêmes 4 096 tokens.

DURÉE : réglée par N_ETAPES (20 000 étapes ≈ 1,2 passage sur le corpus). Ctrl-C
arrête proprement : le dernier modèle est sauvé, on peut reprendre (REPRENDRE).
MAX_MINUTES sert à fixer un budget de temps.

Fichiers écrits :
  models/checkpoints/{NOM}_meilleur.pt   meilleure validation
  models/checkpoints/{NOM}_dernier.pt    dernier état (avec l'optimiseur : reprise possible)
  models/checkpoints/{NOM}_reglages.json tous les réglages du modèle (sa « fiche d'identité »)
  logs/entrainement_{NOM}.csv            étape, perte train, perte validation, taux
(NOM sépare les expériences : la référence a écrit livres_meilleur.pt, l'expérience B
 écrit livres_T256_meilleur.pt. Rien n'est écrasé : c'est l'historique des modèles.)
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

CHAPITRE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPITRE / "tools"))
sys.path.insert(0, str(CHAPITRE.parent / "04_training" / "tools"))
from bpe_rapide import BPE  # noqa: E402
from checkpoint import DOSSIER_CHECKPOINTS, DOSSIER_LOGS, sauver  # noqa: E402
from donnees_corpus import DOSSIER_TOKENISE, charger_bin, tirer_batch  # noqa: E402
from gpt_torch import GPT, perte  # noqa: E402

# --- réglages (une seule variable à la fois quand on compare des expériences) ---
N_FUSIONS = 4000
D, N_TETES, N_BLOCS, T = 384, 6, 8, 256  # comme P4 (8 blocs)
TAILLE_BATCH = 32  # comme la référence : 8 192 tokens par étape avec T = 256, donc 2,5 passages
N_ETAPES = 20_000
LR_MAX, LR_MIN, WARMUP = 6e-4, 6e-5, 300  # comme A
INTERVALLE_MESURE = 500
MAX_MINUTES = None
REPRENDRE = False  # True : repart de {NOM}_dernier.pt
GRAINE = 0
DONNEES_TRAIN = "train_theatre"  # "train" : livres seuls ; "train_v2" : texte filtré de D (leçons 11-12) ; "train_theatre" : livres + théâtre (leçons 16-17)
DONNEES_DIALOGUES = "theatre_validation"  # None : pas de seconde mesure ; sinon une perte de plus, sur ces dialogues jamais appris (leçon 17)
NOM = "livres_P4_theatre"  # préfixe des fichiers écrits ; chaque expérience a le sien (la référence : "livres")

appareil = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(GRAINE)
rng = np.random.default_rng(GRAINE)

bpe = BPE.charger(DOSSIER_TOKENISE / f"bpe_{N_FUSIONS}.json")
entrainement, validation = charger_bin(DONNEES_TRAIN), charger_bin("validation")
dialogues = charger_bin(DONNEES_DIALOGUES) if DONNEES_DIALOGUES else None
config = dict(taille_vocab=bpe.taille, d=D, n_tetes=N_TETES, n_blocs=N_BLOCS, T_max=T)
modele = GPT(**config).to(appareil)
optimiseur = torch.optim.AdamW(modele.parameters(), lr=LR_MAX, weight_decay=0.1)
chemin_meilleur = DOSSIER_CHECKPOINTS / f"{NOM}_meilleur.pt"
chemin_dernier = DOSSIER_CHECKPOINTS / f"{NOM}_dernier.pt"

print("ÉTAPE 1 — le modèle")
print(f"  appareil : {appareil}" + (f" ({torch.cuda.get_device_name(0)})" if appareil == "cuda" else ""))
print(f"  {modele.nb_parametres():,} paramètres (dont {2 * bpe.taille * D:,} dans les embeddings)")
print(f"  {len(entrainement):,} tokens d'entraînement ({DONNEES_TRAIN}), {len(validation):,} de validation")
if dialogues is not None:
    print(f"  + {len(dialogues):,} tokens de dialogues de validation ({DONNEES_DIALOGUES}), mesurés à part")
print(f"  perte d'un modèle au hasard : ln({bpe.taille}) = {math.log(bpe.taille):.2f}")
tokens_par_etape = TAILLE_BATCH * T
print(f"  {tokens_par_etape:,} tokens par étape -> {N_ETAPES * tokens_par_etape / len(entrainement):.2f} passage(s) sur le corpus")

etape_depart = 0
meilleure = float("inf")
if REPRENDRE and chemin_dernier.exists():
    contenu = torch.load(chemin_dernier)
    modele.load_state_dict(contenu["poids"])
    optimiseur.load_state_dict(contenu["optimiseur"])
    etape_depart, meilleure = contenu["etape"], contenu.get("meilleure", contenu["perte_val"])
    print(f"  reprise à l'étape {etape_depart}")


def taux(etape):
    """Échauffement linéaire, puis décroissance en cosinus de LR_MAX à LR_MIN."""
    if etape < WARMUP:
        return LR_MAX * (etape + 1) / WARMUP
    progres = (etape - WARMUP) / max(1, N_ETAPES - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progres))


def perte_validation(n_batchs=40, donnees=None):
    donnees = validation if donnees is None else donnees
    gen = np.random.default_rng(123)  # mêmes fenêtres à chaque mesure : courbes comparables
    modele.eval()
    with torch.no_grad():
        total = sum(perte(modele, *tirer_batch(donnees, TAILLE_BATCH, T, gen, appareil)).item()
                    for _ in range(n_batchs))
    modele.train()
    return total / n_batchs


def echantillon(debut="Il était une fois", n=50):
    ids = torch.tensor([bpe.encoder(debut)], device=appareil)
    return bpe.decoder(modele.generer(ids, n, temperature=0.8)[0].tolist())


# fiche d'identité du modèle : le checkpoint contient d, blocs, T ; ici on garde TOUS les réglages
DOSSIER_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
reglages = dict(nom=NOM, **config, parametres=modele.nb_parametres(), taille_batch=TAILLE_BATCH,
                n_etapes=N_ETAPES, lr_max=LR_MAX, lr_min=LR_MIN, warmup=WARMUP, n_fusions=N_FUSIONS, graine=GRAINE)
(DOSSIER_CHECKPOINTS / f"{NOM}_reglages.json").write_text(json.dumps(reglages, indent=2, ensure_ascii=False))

print("\nÉTAPE 2 — entraînement")
DOSSIER_LOGS.mkdir(exist_ok=True)
journal = open(DOSSIER_LOGS / f"entrainement_{NOM}.csv", "a" if REPRENDRE else "w", newline="")
ecrivain = csv.writer(journal)
if not REPRENDRE:
    ecrivain.writerow(["etape", "perte_train", "perte_val", "taux"] + (["perte_dialogues"] if dialogues is not None else []))

debut_temps = time.time()
somme, compte = 0.0, 0
etape = etape_depart
try:
    for etape in range(etape_depart + 1, N_ETAPES + 1):
        for groupe in optimiseur.param_groups:
            groupe["lr"] = taux(etape)
        x, y = tirer_batch(entrainement, TAILLE_BATCH, T, rng, appareil)
        l = perte(modele, x, y)
        optimiseur.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optimiseur.step()
        somme, compte = somme + l.item(), compte + 1

        fin = MAX_MINUTES is not None and (time.time() - debut_temps) / 60 >= MAX_MINUTES
        if etape % INTERVALLE_MESURE == 0 or etape == N_ETAPES or fin:
            pt, pv = somme / compte, perte_validation()
            somme, compte = 0.0, 0
            ecoule = time.time() - debut_temps
            restant = ecoule / (etape - etape_depart) * (N_ETAPES - etape)
            marque = ""
            if pv < meilleure:
                meilleure, marque = pv, "  <- record"
                sauver(chemin_meilleur, modele, config, etape, pv, optimiseur)
            pd = perte_validation(donnees=dialogues) if dialogues is not None else None
            ecrivain.writerow([etape, f"{pt:.4f}", f"{pv:.4f}", f"{taux(etape):.6f}"] + ([f"{pd:.4f}"] if pd is not None else []))
            journal.flush()
            texte_dialogues = f"  dialogues {pd:.3f}" if pd is not None else ""
            print(f"  étape {etape:>6} : train {pt:.3f}  val {pv:.3f} (perplexité {math.exp(pv):.0f}){texte_dialogues}"
                  f"  | {ecoule / 60:.0f} min, reste ~{restant / 60:.0f} min{marque}")
            print(f"      « {echantillon()!r} »")
            if fin:
                print("  budget de temps atteint, arrêt.")
                break
except KeyboardInterrupt:
    print("\n  interrompu (Ctrl-C) : on sauve l'état courant.")

journal.close()
sauver(chemin_dernier, modele, config, etape, perte_validation(), optimiseur)
contenu = torch.load(chemin_dernier)
contenu["meilleure"] = meilleure
torch.save(contenu, chemin_dernier)

print("\nÉTAPE 3 — bilan")
print(f"  meilleure validation : {meilleure:.3f} (perplexité {math.exp(meilleure):.0f}), contre {math.log(bpe.taille):.2f} au hasard")
print(f"  checkpoints : {chemin_meilleur.name}, {chemin_dernier.name} (dans models/checkpoints/)")
print(f"  courbe : logs/entrainement_{NOM}.csv")
