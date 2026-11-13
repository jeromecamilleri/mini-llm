"""Évaluation d'un agent Sudoku — evaluation.py

Outils de la leçon 03 : masque des actions autorisées, une partie complète,
et mesure d'une politique sur plusieurs grilles.

Une politique est une fonction  politique(env) -> action  (entier de 0 à 728).
Elle remplacera plus tard le réseau : la mesure restera la même.
"""
import numpy as np

from sudoku_utils import N_ACTIONS


def masque_actions(grille: np.ndarray) -> np.ndarray:
    """Masque booléen (729,) : True si l'action est autorisée.

    Autorisée = la case est vide ET le chiffre ne crée pas de doublon
    dans la ligne, la colonne ou le bloc 3×3. Ne dépend pas de la solution :
    le réseau pourra l'utiliser sans rien savoir de l'oracle.
    """
    interdit = np.zeros((9, 9, 10), dtype=bool)  # [ligne, col, chiffre]
    for i in range(9):
        for k in range(9):
            v = grille[i, k]
            if v:
                interdit[i, :, v] = True
                interdit[:, k, v] = True
                r0, c0 = 3 * (i // 3), 3 * (k // 3)
                interdit[r0:r0 + 3, c0:c0 + 3, v] = True
    interdit[grille != 0, :] = True
    return ~interdit[:, :, 1:].reshape(N_ACTIONS)


def politique_hasard(env, rng: np.random.Generator) -> int:
    """Une action au hasard parmi les 729, sans aucune règle."""
    return int(rng.integers(N_ACTIONS))


def politique_hasard_masque(env, rng: np.random.Generator) -> int:
    """Une action au hasard parmi les actions autorisées (case vide, pas de doublon)."""
    autorisees = np.flatnonzero(masque_actions(env.grille))
    return int(rng.choice(autorisees))


def politique_case_contrainte(env, rng: np.random.Generator) -> int:
    """Choisit la case vide qui a le moins de chiffres autorisés, puis un chiffre au hasard parmi eux.

    Référence « humaine » : quand une case n'accepte qu'un chiffre, ce coup est sûr.
    N'utilise pas la solution, seulement les règles.
    """
    autorisees = masque_actions(env.grille).reshape(81, 9)
    n_par_case = autorisees.sum(axis=1)
    n_par_case = np.where(n_par_case == 0, 99, n_par_case)
    case = int(rng.choice(np.flatnonzero(n_par_case == n_par_case.min())))
    chiffre = int(rng.choice(np.flatnonzero(autorisees[case])))
    return case * 9 + chiffre


def jouer_episode(env, politique) -> dict:
    """Joue une partie jusqu'à la fin et retourne ses statistiques."""
    env.reset()
    retour = 0.0
    n_oui = 0
    fini = False
    while not fini:
        _, r, fini, info = env.step(politique(env))
        retour += r
        n_oui += info["oracle_verdict"] == "OUI"
    return {
        "retour": retour,
        "coups": env.n_coups,
        "oui": n_oui,
        "resolue": env.n_cases_vides == 0,
    }


def evaluer(politique_creer, dataset, n_grilles: int, n_coups_max: int, graine: int = 0) -> dict:
    """Mesure une politique sur les n premières grilles du dataset.

    politique_creer(rng) doit retourner une politique (env) -> action.
    Retourne le taux de grilles résolues, le retour moyen, la part de coups
    corrects et le nombre moyen de coups des grilles résolues.
    """
    from sudoku_env import SudokuEnv

    rng = np.random.default_rng(graine)
    politique = politique_creer(rng)
    stats = []
    for i in range(n_grilles):
        item = dataset[i]
        env = SudokuEnv(item["puzzle"], item["solution"], n_coups_max=n_coups_max)
        stats.append(jouer_episode(env, politique))
    resolues = [s for s in stats if s["resolue"]]
    return {
        "n": n_grilles,
        "taux_resolu": len(resolues) / n_grilles,
        "retour_moyen": float(np.mean([s["retour"] for s in stats])),
        "part_oui": float(sum(s["oui"] for s in stats) / sum(s["coups"] for s in stats)),
        "coups_si_resolue": float(np.mean([s["coups"] for s in resolues])) if resolues else float("nan"),
    }
