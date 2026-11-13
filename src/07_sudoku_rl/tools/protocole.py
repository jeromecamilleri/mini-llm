"""Protocole d'évaluation de référence — protocole.py

Phase 1 du plan (docs/plan_agent_sudoku_generique_sans_labels.md) : mesurer tous les agents
de la même façon, sur des grilles que l'entraînement n'a JAMAIS vues.

Jeu de test indépendant
    512 grilles tirées dans le dataset complet (200 000 grilles), 128 par niveau de difficulté,
    en excluant les 1 000 grilles du mini-dataset (le « pool » d'entraînement des essais 1 à 15).
    Leurs identifiants sont écrits dans logs/sudoku_test_ids.csv (graine 12345).

Ce qu'on compte (définitions)
    coups      : nombre d'actions jouées, les coups refusés comptés ; une partie non résolue
                 compte l'horizon (150) ; les actions interdites par le masque n'existent pas.
    faux       : coups refusés par l'environnement (case vide mais mauvais chiffre).
    répétitions: coups faux qui rejouent un coup déjà refusé sur la même grille.
    retour     : somme des récompenses de la partie.

Trois façons de jouer
    hasard : un coup tiré uniformément parmi les coups autorisés par le masque (= le réseau avant
             apprentissage) ; référence basse.
    mrv    : la case qui a le moins de chiffres légaux (égalité : au hasard), puis un chiffre légal
             au hasard. Référence ALGORITHMIQUE externe : jamais une cible d'apprentissage.
    réseau : un coup tiré selon les probabilités du réseau. Sans mémoire des refus.

Quatrième agent sans apprentissage (évaluation A détaillée, guide de décision Phase 1)
    hasard factorisé : une case tirée uniformément parmi celles qui ont un chiffre légal, puis un
             chiffre légal uniforme. Mêmes règles que `hasard`, mais autre distribution (voir
             docs/audits/contre_expertise_phase0_sudoku_rl_2026-11-13_sonnet.md, §4).
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env_lots import SudokuEnvLots

RACINE = Path(__file__).resolve().parents[3]
DATASET = RACINE / "datasets" / "sudoku" / "sudoku_dataset.csv"
MINI = RACINE / "datasets" / "sudoku" / "sudoku_dataset_mini.csv"
IDS_TEST = RACINE / "logs" / "sudoku_test_ids.csv"
NIVEAUX = ("facile", "moyen", "difficile", "diabolique")


def _lire(chemin):
    with open(chemin) as f:
        return list(csv.DictReader(f))


def _grilles(lignes, cle):
    return np.array([[int(c) for c in l[cle]] for l in lignes], dtype=np.int64).reshape(-1, 9, 9)


def construire_test(par_niveau=128, graine=12345):
    """Tire le jeu de test hors du pool d'entraînement et écrit ses identifiants."""
    complet = _lire(DATASET)
    pool = {l["puzzle"] for l in _lire(MINI)}
    rng = np.random.default_rng(graine)
    ids = []
    for niveau in NIVEAUX:
        candidats = [i for i, l in enumerate(complet) if l["difficulte"] == niveau and l["puzzle"] not in pool]
        ids += sorted(int(i) for i in rng.choice(candidats, size=par_niveau, replace=False))
    with open(IDS_TEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "difficulte"])
        for i in ids:
            w.writerow([i, complet[i]["difficulte"]])
    return ids


def ids_test():
    """Identifiants (lignes du dataset complet) réservés au test."""
    with open(IDS_TEST) as f:
        return [int(l["id"]) for l in csv.DictReader(f)]


def charger_test():
    """Retourne puzzles (N,9,9), solutions (N,9,9), difficultés (liste)."""
    complet = _lire(DATASET)
    lignes = [complet[i] for i in ids_test()]
    return _grilles(lignes, "puzzle"), _grilles(lignes, "solution"), [l["difficulte"] for l in lignes]


# ── politiques : (env, generateur) → actions (B,) ──────────────────────────────────────────

def _tirer(poids, gen):
    """Tire une action par ligne ; une ligne sans poids (partie terminée) joue 0."""
    vide = ~(poids > 0).any(dim=1)
    poids = torch.where(vide[:, None], torch.ones_like(poids), poids)
    return torch.multinomial(poids, 1, generator=gen).squeeze(1)


def politique_hasard(env, gen):
    return _tirer(env.masques().float(), gen)


def politique_mrv(env, gen):
    m = env.masques().reshape(env.B, 81, 9)
    n = m.sum(dim=2).float()
    score = torch.where(n > 0, n + 0.5 * torch.rand(n.shape, device=n.device, generator=gen), torch.full_like(n, 1e9))
    case = score.argmin(dim=1)
    chiffres = m[env._idx, case].float()
    return case * 9 + _tirer(chiffres, gen)


def politique_reseau(modele):
    def politique(env, gen):
        logits, _ = modele(env.etats(modele.voit_refus), env.masques())
        return torch.multinomial(F.softmax(logits, dim=1), 1, generator=gen).squeeze(1)
    return politique


@torch.no_grad()
def jouer_protocole(politique, puzzles, solutions, horizon, appareil, graine):
    """Joue chaque grille jusqu'à la fin ou à l'horizon. Retourne un dictionnaire de tableaux (B,)."""
    env = SudokuEnvLots(puzzles, solutions, n_coups_max=horizon, appareil=appareil, memoire=False)
    gen = torch.Generator(device=appareil).manual_seed(graine)
    faux = torch.zeros(env.B, device=appareil)
    repetitions = torch.zeros(env.B, device=appareil)
    retour = torch.zeros(env.B, device=appareil)
    while not bool(env.termine.all()):
        a = politique(env, gen)
        actif = ~env.termine
        deja = env.refuse[env._idx, a]
        _, r, _, ok = env.step(a)
        mal = actif & ~ok
        faux += mal
        repetitions += mal & deja
        retour += r
    return {"resolue": (env.n_vides == 0).float().cpu().numpy(), "coups": env.n_coups.float().cpu().numpy(),
            "faux": faux.cpu().numpy(), "repetitions": repetitions.cpu().numpy(), "retour": retour.cpu().numpy()}


def resumer(res, difficultes):
    """Moyennes par niveau et au total : {niveau ou 'tous': {mesure: valeur}}."""
    diff = np.array(difficultes)
    sortie = {}
    for nom, masque in [("tous", np.ones(len(diff), bool))] + [(n, diff == n) for n in NIVEAUX]:
        sortie[nom] = {k: float(v[masque].mean()) for k, v in res.items()}
    return sortie


# ── évaluation A détaillée : diagnostics par k (guide de décision, Phase 1) ─────────────────
# k = nombre de chiffres légaux dans la case jouée. Un seul chiffre légal (k = 1) = coup toujours juste
# (la solution est unique). Ces mesures n'entrent JAMAIS dans l'entraînement : elles décrivent seulement.

KMAX = 5  # k = 1, 2, 3, 4, puis « 5 et plus »


def politique_hasard_factorise(env, gen):
    """Une case uniforme parmi celles qui ont un chiffre légal, puis un chiffre légal uniforme."""
    m = env.masques().reshape(env.B, 81, 9)
    case = _tirer(m.any(dim=2).float(), gen)
    return case * 9 + _tirer(m[env._idx, case].float(), gen)


def politique_reseau_entropie(modele):
    """Comme politique_reseau, et garde l'entropie jointe de la politique à chaque coup (politique.derniere_entropie)."""
    def politique(env, gen):
        logits, _ = modele(env.etats(modele.voit_refus), env.masques())
        lp = F.log_softmax(logits, dim=1)
        politique.derniere_entropie = -(lp.exp() * lp.clamp(min=-30)).sum(dim=1)
        return torch.multinomial(F.softmax(logits, dim=1), 1, generator=gen).squeeze(1)
    return politique


@torch.no_grad()
def jouer_detail(politique, puzzles, solutions, horizon, appareil, graine):
    """Comme jouer_protocole, avec en plus par grille (B,) : coups et coups faux selon k
    (coups_k1..coups_k4, coups_k5p ; faux_k1..), nombre de coups joués alors qu'une case à chiffre
    unique existait (k1_dispo) et parmi eux ceux qui ont choisi une telle case (choix_k1),
    et l'entropie jointe moyenne par coup (nan si la politique ne la fournit pas)."""
    env = SudokuEnvLots(puzzles, solutions, n_coups_max=horizon, appareil=appareil, memoire=False)
    gen = torch.Generator(device=appareil).manual_seed(graine)
    z = lambda: torch.zeros(env.B, device=appareil)
    faux, repetitions, retour, k1_dispo, choix_k1, entropie, n_actifs = z(), z(), z(), z(), z(), z(), z()
    coups_k, faux_k = [z() for _ in range(KMAX)], [z() for _ in range(KMAX)]
    avec_entropie = False
    while not bool(env.termine.all()):
        k = env.masques().reshape(env.B, 81, 9).sum(dim=2)
        a = politique(env, gen)
        actif = ~env.termine
        kc = k[env._idx, a // 9]
        deja = env.refuse[env._idx, a]
        dispo = (k == 1).any(dim=1)
        _, r, _, ok = env.step(a)
        mal = actif & ~ok
        faux += mal
        repetitions += mal & deja
        retour += r
        n_actifs += actif
        k1_dispo += actif & dispo
        choix_k1 += actif & dispo & (kc == 1)
        e = getattr(politique, "derniere_entropie", None)
        if e is not None:
            avec_entropie = True
            entropie += torch.where(actif, e, torch.zeros_like(e))
        kbin = kc.clamp(min=1, max=KMAX)
        for j in range(KMAX):
            coups_k[j] += actif & (kbin == j + 1)
            faux_k[j] += mal & (kbin == j + 1)
    sortie = {"resolue": (env.n_vides == 0).float(), "coups": env.n_coups.float(), "faux": faux,
              "repetitions": repetitions, "retour": retour, "k1_dispo": k1_dispo, "choix_k1": choix_k1}
    for j in range(KMAX):
        nom = f"k{j + 1}" if j < KMAX - 1 else f"k{KMAX}p"
        sortie[f"coups_{nom}"], sortie[f"faux_{nom}"] = coups_k[j], faux_k[j]
    sortie["entropie"] = entropie / n_actifs.clamp(min=1) if avec_entropie else torch.full_like(faux, float("nan"))
    return {c: v.cpu().numpy() for c, v in sortie.items()}
