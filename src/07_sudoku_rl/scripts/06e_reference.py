"""
RÉFÉRENCE (phase 1 du plan) : mesurer chaque agent sur des grilles jamais vues
===============================================================================

    # une seule fois : tirer le jeu de test indépendant (512 grilles, 128 par difficulté)
    .venv/bin/python src/07_sudoku_rl/scripts/06e_reference.py --construire

    # mesurer : « hasard » et « mrv » sont des agents sans réseau, le reste des checkpoints
    .venv/bin/python src/07_sudoku_rl/scripts/06e_reference.py hasard mrv \\
        sudoku_reference_g0 sudoku_reference_g1 sudoku_reference_g2 sudoku_reference_g3

POURQUOI
Jusqu'ici, les grilles de test venaient du même mini-dataset de 1 000 grilles que
l'entraînement : un réseau pouvait un peu les « connaître ». Ici les 512 grilles de test sont
exclues de l'entraînement. Chaque agent joue ces grilles avec 4 tirages différents (graines
d'échantillonnage 0 à 3) ; les définitions de « coups », « faux » et « répétitions » sont dans
tools/protocole.py. On ne donne jamais la solution au réseau, et le masque ne dépend que des
règles (ligne, colonne, bloc).

Avec plusieurs checkpoints (une graine d'entraînement chacun), on affiche en plus la moyenne et
l'écart-type ENTRE ENTRAÎNEMENTS : c'est cette dispersion qui dit si une différence est réelle.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "src" / "07_sudoku_rl" / "tools"))

import protocole as P
from reseau import charger_modele

p = argparse.ArgumentParser()
p.add_argument("agents", nargs="*", help="hasard, mrv, ou nom d'un checkpoint (sans .pt)")
p.add_argument("--construire", action="store_true", help="tire le jeu de test et écrit logs/sudoku_test_ids.csv")
p.add_argument("--horizon", type=int, default=150)
p.add_argument("--appareil", default="cuda")
args = p.parse_args()

if args.construire:
    ids = P.construire_test()
    print(f"{len(ids)} grilles de test écrites dans {P.IDS_TEST.relative_to(RACINE)}")
    sys.exit()
if not args.agents:
    sys.exit("donner au moins un agent (hasard, mrv ou un checkpoint), ou --construire")

puz, sol, diff = P.charger_test()
print(f"jeu de test : {len(puz)} grilles indépendantes, horizon {args.horizon}, 4 tirages par agent\n")
MESURES = ("resolue", "coups", "faux", "repetitions", "retour")
EN_TETE = f"  {'':11s} {'résolues':>9s} {'coups':>7s} {'faux':>7s} {'répét.':>7s} {'retour':>8s}"


def ligne(nom, r):
    return (f"  {nom:11s} {r['resolue']:9.1%} {r['coups']:7.1f} {r['faux']:7.1f} "
            f"{r['repetitions']:7.1f} {r['retour']:+8.1f}")


par_agent = {}
for agent in args.agents:
    if agent == "hasard":
        politique = P.politique_hasard
    elif agent == "mrv":
        politique = P.politique_mrv
    else:
        politique = P.politique_reseau(charger_modele(RACINE / "models" / "checkpoints" / f"{agent}.pt", args.appareil).eval())
    resumes = [P.resumer(P.jouer_protocole(politique, puz, sol, args.horizon, args.appareil, g), diff) for g in range(4)]
    moyen = {niv: {k: float(np.mean([r[niv][k] for r in resumes])) for k in MESURES} for niv in resumes[0]}
    par_agent[agent] = moyen
    print(f"agent : {agent}  (moyenne de 4 tirages)")
    print(EN_TETE)
    for niv in ("tous",) + P.NIVEAUX:
        print(ligne(niv, moyen[niv]))
    print()

reseaux = [a for a in args.agents if a not in ("hasard", "mrv")]
if len(reseaux) > 1:
    print(f"ENTRE ENTRAÎNEMENTS ({len(reseaux)} checkpoints) : moyenne ± écart-type, toutes difficultés")
    for k, nom in zip(MESURES, ("résolues", "coups", "faux", "répétitions", "retour")):
        v = np.array([par_agent[a]["tous"][k] for a in reseaux])
        fmt = "{:.1%} ± {:.1%}" if k == "resolue" else "{:.1f} ± {:.1f}"
        print(f"  {nom:12s}: " + fmt.format(v.mean(), v.std(ddof=1)) + "   (" + ", ".join(
            (f"{x:.1%}" if k == "resolue" else f"{x:.1f}") for x in v) + ")")
