"""
RÉFÉRENCES A (guide de décision, Phase 1) : environnement AVEC oracle, résultats détaillés
==========================================================================================

    .venv/bin/python src/07_sudoku_rl/scripts/06f_references_A.py hasard hasard_facto mrv \\
        sudoku_reference_g0 sudoku_reference_g1 sudoku_reference_g2 sudoku_reference_g3 \\
        | tee logs/sudoku_reference_A_sortie.txt

Reprend le protocole de 06e (512 grilles de test hors pool, 4 tirages par agent, horizon 150, sans
mémoire des refus) et ajoute :
    - le « hasard factorisé » (case uniforme, puis chiffre légal uniforme) ;
    - les résultats grille par grille et tirage par tirage : logs/sudoku_reference_A_par_grille.csv ;
    - des diagnostics selon k (nombre de chiffres légaux dans la case jouée) : part des coups,
      taux de coups faux, choix d'une case à chiffre unique quand il en existe une, entropie jointe ;
    - la configuration (commit, appareil, empreintes SHA-256 des données et des checkpoints).

Ces mesures DÉCRIVENT l'agent ; aucune n'entre dans l'entraînement. Le résultat « moyenne ± écart-type
entre entraînements » compte les checkpoints (une graine d'entraînement chacun), pas les tirages :
plusieurs tirages sur la même grille ne sont pas des grilles indépendantes.

Un checkpoint qui voit les refus (19 canaux) est refusé : la mémoire des refus est exclue du projet.
L'appareil par défaut est le CPU : les nombres tirés dépendent de l'appareil, donc les chiffres
diffèrent un peu d'un appareil à l'autre (hasard plat : 91,4 coups sur CPU, 91,6 sur GPU).
"""
import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

import protocole as P
from reseau import charger_modele

p = argparse.ArgumentParser()
p.add_argument("agents", nargs="+", help="hasard, hasard_facto, mrv, ou nom d'un checkpoint (sans .pt)")
p.add_argument("--horizon", type=int, default=150)
p.add_argument("--tirages", type=int, default=4)
p.add_argument("--appareil", default="cpu")
p.add_argument("--csv", default=str(RACINE / "logs" / "sudoku_reference_A_par_grille.csv"))
args = p.parse_args()


def sha(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()[:16]


def git(*a):
    return subprocess.run(["git", *a], cwd=RACINE, capture_output=True, text=True).stdout.strip()


CHECKPOINTS = RACINE / "models" / "checkpoints"
print("CONFIGURATION")
print(f"  commit {git('rev-parse', '--short', 'HEAD')} | fichiers modifiés non commités : {len(git('status', '--porcelain').splitlines())}")
print(f"  appareil {args.appareil} | horizon {args.horizon} | {args.tirages} tirages | torch {torch.__version__}")
for nom, chemin in (("dataset complet", P.DATASET), ("mini-pool", P.MINI), ("ids de test", P.IDS_TEST)):
    print(f"  sha256 {nom:16s}: {sha(chemin)}")

puz, sol, diff = P.charger_test()
ids = P.ids_test()
print(f"  jeu de test : {len(puz)} grilles indépendantes")

POLITIQUES = {"hasard": P.politique_hasard, "hasard_facto": P.politique_hasard_factorise, "mrv": P.politique_mrv}
MESURES = ("resolue", "coups", "faux", "repetitions", "retour")
KS = ("k1", "k2", "k3", "k4", "k5p")
COLONNES = (*MESURES, "k1_dispo", "choix_k1", *(f"coups_{k}" for k in KS), *(f"faux_{k}" for k in KS), "entropie")

resultats = {}   # agent -> liste (un par tirage) de dictionnaires de tableaux (512,)
for agent in args.agents:
    if agent in POLITIQUES:
        politique = POLITIQUES[agent]
    else:
        chemin = CHECKPOINTS / f"{agent}.pt"
        print(f"  sha256 checkpoint {agent}: {sha(chemin)}")
        modele = charger_modele(chemin, args.appareil).eval()
        if modele.voit_refus:
            sys.exit(f"{agent} voit les refus (19 canaux) : exclu du protocole (pas de mémoire des refus)")
        politique = P.politique_reseau_entropie(modele)
    resultats[agent] = [P.jouer_detail(politique, puz, sol, args.horizon, args.appareil, g) for g in range(args.tirages)]

with open(args.csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["agent", "tirage", "id_grille", "difficulte", *COLONNES])
    for agent, tirages in resultats.items():
        for t, r in enumerate(tirages):
            for i in range(len(ids)):
                w.writerow([agent, t, ids[i], diff[i]] + [f"{r[c][i]:.4g}" for c in COLONNES])
print(f"\nrésultats par grille : {args.csv}")


def moyenne(agent, cle, masque=None):
    """Moyenne sur les grilles (et les tirages) ; masque (512,) bool pour une difficulté."""
    v = np.stack([r[cle] for r in resultats[agent]])
    return float(v[:, masque].mean() if masque is not None else v.mean())


def part(agent, num, den):
    """Total(num) / total(den) sur toutes les grilles et tous les tirages."""
    n = sum(float(r[num].sum()) for r in resultats[agent])
    d = sum(float(r[den].sum()) for r in resultats[agent])
    return n / d if d else float("nan")


diff = np.array(diff)
print("\n" + "=" * 78 + "\nA. RÉSULTATS (moyenne de " + f"{args.tirages} tirages)\n" + "=" * 78)
print(f"  {'agent':22s} {'niveau':11s} {'résolues':>9s} {'coups':>7s} {'faux':>7s} {'répét.':>7s} {'retour':>8s}")
for agent in resultats:
    for niveau, masque in (("tous", None), *((n, diff == n) for n in P.NIVEAUX)):
        v = [moyenne(agent, c, masque) for c in MESURES]
        print(f"  {agent if niveau == 'tous' else '':22s} {niveau:11s} {v[0]:9.1%} {v[1]:7.1f} {v[2]:7.1f} {v[3]:7.1f} {v[4]:+8.1f}")

print("\n" + "=" * 78 + "\nB. DIAGNOSTICS SELON k (tous niveaux ; k = chiffres légaux dans la case jouée)\n" + "=" * 78)
print(f"  {'agent':22s} " + " ".join(f"{'faux ' + k:>8s}" for k in KS) + f" {'P(k=1|dispo)':>13s} {'coups dispo':>12s} {'entropie':>9s}")
print(f"  {'taux de hasard (1-1/k)':22s} " + " ".join(f"{1 - 1 / j:8.2f}" for j in (1, 2, 3, 4)) + f" {'≥ 0.80':>8s}")
for agent in resultats:
    faux = " ".join(f"{part(agent, f'faux_{k}', f'coups_{k}'):8.2f}" for k in KS)
    dispo = part(agent, "k1_dispo", "coups")
    e = moyenne(agent, "entropie")
    print(f"  {agent:22s} {faux} {part(agent, 'choix_k1', 'k1_dispo'):13.3f} {dispo:12.3f} {'' if np.isnan(e) else f'{e:9.3f}'}")
print("  (faux k1 = 0 pour tout agent : un chiffre unique est toujours juste, c'est le masque qui l'impose ;")
print("   P(k=1|dispo) = part des coups qui jouent une case à chiffre unique quand il en existe une ;")
print("   coups dispo = part des coups joués alors qu'une telle case existait)")

reseaux = [a for a in resultats if a not in POLITIQUES]
if len(reseaux) > 1:
    print("\n" + "=" * 78 + f"\nC. ENTRE ENTRAÎNEMENTS ({len(reseaux)} checkpoints) : moyenne ± écart-type (n-1)\n" + "=" * 78)
    for nom, f in (("résolues", lambda a: moyenne(a, "resolue")), ("coups", lambda a: moyenne(a, "coups")),
                   ("faux", lambda a: moyenne(a, "faux")), ("répétitions", lambda a: moyenne(a, "repetitions")),
                   ("retour", lambda a: moyenne(a, "retour")), ("P(k=1|dispo)", lambda a: part(a, "choix_k1", "k1_dispo")),
                   ("entropie", lambda a: moyenne(a, "entropie"))):
        v = np.array([f(a) for a in reseaux])
        fmt = "{:.1%} ± {:.1%}" if nom == "résolues" else "{:.3f} ± {:.3f}" if nom in ("P(k=1|dispo)", "entropie") else "{:.1f} ± {:.1f}"
        print(f"  {nom:14s}: " + fmt.format(v.mean(), v.std(ddof=1)) + "   (" + ", ".join(
            f"{x:.1%}" if nom == "résolues" else f"{x:.3f}" if nom in ("P(k=1|dispo)", "entropie") else f"{x:.1f}" for x in v) + ")")
