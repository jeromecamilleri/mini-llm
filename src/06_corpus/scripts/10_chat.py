"""06_corpus — Leçon 10 : parler au modèle (chat simplifié) et comprendre entraînement / utilisation.

    .venv/bin/python src/06_corpus/scripts/10_chat.py [fichier.pt]

Sans argument, le chat charge CHECKPOINT (D) (réglage ci-dessous). Pour un essai à la main
sur un autre modèle (ex. P1) : ... 10_chat.py livres_P1_6blocs_meilleur.pt

DEUX ÉTAPES DIFFÉRENTES, MÊME MODÈLE
                      ENTRAÎNEMENT (leçon 7)           UTILISATION (ici)
  but                 apprendre les poids              se servir des poids appris
  poids               changent à chaque étape          figés (modele.eval(), pas de gradient)
  texte donné         des milliers de fenêtres         ton texte, une seule fois
                      tirées des livres
  réponse attendue    connue : le token suivant        inconnue : le modèle la devine
                      réel du livre
  perte, AdamW        oui                              non (rien n'est corrigé)
  coût                14 minutes de GPU                quelques secondes de CPU
  résultat            un fichier .pt                   du texte

APPAREIL : le chat tourne par défaut sur le CPU, car une passe avant sur un si petit
modèle est très rapide. Mettre APPAREIL = "cuda" pour comparer : le temps affiché
après chaque réponse ne diminue pas forcément (le calcul est trop petit pour le GPU).

Pendant l'utilisation, le modèle ne retient RIEN de ta conversation : à chaque
token, il relit seulement les 128 derniers tokens et prédit le suivant. Ce qu'il
a appris est dans les poids, rien n'y est ajouté.

CE N'EST PAS UN VRAI ASSISTANT. Il a seulement appris à continuer des livres, pas à
répondre à des questions. Deux modes :

  chat  : ta phrase devient une réplique de dialogue de roman (« - ta phrase »),
          et le modèle écrit la réplique suivante (il s'arrête à la fin de la ligne).
          La conversation s'accumule dans la fenêtre de 128 tokens : au-delà,
          les plus anciennes répliques sont oubliées.
  suite : le modèle continue ton texte, sur la longueur choisie.

Commandes (taper « / » puis le nom) :
  /chat  /suite          changer de mode
  /temp 0.8              température (basse = prudent, haute = audacieux)
  /topk 40  /topk 0      Top-K (0 = désactivé)
  /topp 0.9 /topp 0      Top-P (0 = désactivé)
  /longueur 80           nombre maximal de tokens produits
  /nouveau               oublier la conversation
  /memoire 0  /memoire 1 sans mémoire (chaque question est indépendante) / avec mémoire

Pourquoi une option sans mémoire ? Le modèle recopie ce qu'il vient de lire,
y compris ses propres réponses. Mesuré sur le modèle A (température 0,6, Top-K 40) :
  « je te dis » après « pourquoi » (conversation neuve) : 1 chance sur 10 000
  « je te dis » après « il fait beau » (conversation neuve) : 1 sur 20 000
  « je te dis » après « il fait beau », avec la 1re réponse dans la mémoire : 5,5 %
  « Ô » en premier mot : 1,5 % sans mémoire, 98 % après trois réponses en « Ô ».
Baisser la température aggrave la boucle. Remède : /nouveau ou /memoire 0.
  /quitter               sortir (ou Ctrl-D)
"""

import sys
import time
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
from generation import choisir, filtrer_top_k, filtrer_top_p, probabilites, scores_suivant  # noqa: E402
from gpt_torch import GPT  # noqa: E402

# --- réglages de départ (modifiables avec les commandes) ---
N_FUSIONS = 4000
CHECKPOINT = "livres_D_filtre_meilleur.pt"  # précédents : "livres_A_d384_meilleur.pt" (A), "livres_T256_meilleur.pt" (B), "livres_meilleur.pt" (référence)
APPAREIL = "cpu"  # "cuda" pour utiliser le GPU : comparer le temps affiché après chaque réponse
temperature, top_k, top_p, longueur = 0.8, 40, None, 80
mode = "chat"

bpe = BPE.charger(DOSSIER_TOKENISE / f"bpe_{N_FUSIONS}.json")
if len(sys.argv) > 1:
    CHECKPOINT = sys.argv[1]  # essai à la main : nom d'un fichier de models/checkpoints/
contenu = torch.load(DOSSIER_CHECKPOINTS / CHECKPOINT, map_location=APPAREIL)
modele = GPT(**contenu["config"]).to(APPAREIL)
modele.load_state_dict(contenu["poids"])
modele.eval()  # utilisation : les poids ne bougent plus
T = modele.T_max

print(f"Modèle chargé : {CHECKPOINT} sur {APPAREIL}, {modele.nb_parametres():,} paramètres, "
      f"validation {contenu['perte_val']:.3f}.")
print(f"Il ne lit que les {T} derniers tokens. Tape /quitter pour sortir.")
print("Rappel : il a appris à continuer des romans, pas à répondre à des questions.\n")

memoire = True  # False : on vide l'historique après chaque réponse
historique = []  # identifiants de tokens de la conversation


def produire(ids, arret_ligne):
    """Écrit à l'écran, token par token, la suite de ids. Retourne les tokens produits."""
    nouveaux = []
    for _ in range(longueur):
        probas = probabilites(scores_suivant(modele, ids + nouveaux), temperature)
        if top_k:
            probas = filtrer_top_k(probas, top_k)
        if top_p:
            probas = filtrer_top_p(probas, top_p)
        suivant = choisir(probas, torch.Generator(device=APPAREIL).manual_seed(int(time.time_ns() % 2**31)))
        morceau = bpe.decoder([suivant])
        if arret_ligne and "\n" in morceau:
            nouveaux.append(suivant)
            break
        nouveaux.append(suivant)
        print(morceau, end="", flush=True)
    print()
    return nouveaux


def commande(ligne):
    """Exécute une commande « /nom valeur ». Retourne False pour quitter."""
    global temperature, top_k, top_p, longueur, mode, historique, memoire
    nom, *valeur = ligne[1:].split()
    try:
        if nom in ("quitter", "q"):
            return False
        elif nom in ("chat", "suite"):
            mode = nom
        elif nom == "temp":
            temperature = float(valeur[0])
        elif nom == "topk":
            top_k = int(valeur[0]) or None
        elif nom == "topp":
            top_p = float(valeur[0]) or None
        elif nom == "longueur":
            longueur = int(valeur[0])
        elif nom == "nouveau":
            historique = []
        elif nom == "memoire":
            memoire = bool(int(valeur[0]))
            historique = []
        else:
            print("  commande inconnue (voir le début du fichier)")
            return True
    except (IndexError, ValueError):
        print("  valeur manquante ou invalide")
        return True
    print(f"  mode {mode}, température {temperature}, top_k {top_k}, top_p {top_p}, longueur {longueur}")
    return True


while True:
    try:
        ligne = input(f"[{mode}] > ").strip()
    except EOFError:
        break
    if not ligne:
        continue
    if ligne.startswith("/"):
        if not commande(ligne):
            break
        continue

    debut = time.time()
    if mode == "chat":
        historique += bpe.encoder(f"- {ligne}\n- ")
        print("- ", end="")
        historique += produire(historique, arret_ligne=True)
    else:
        historique += bpe.encoder(ligne)
        print(ligne, end="")
        historique += produire(historique, arret_ligne=False)
    historique = historique[-T:] if memoire else []  # ce qui dépasse la fenêtre est oublié
    print(f"  ({time.time() - debut:.1f} s, {len(historique)}/{T} tokens en mémoire)\n")
