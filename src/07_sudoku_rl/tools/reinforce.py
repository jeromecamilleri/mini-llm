"""REINFORCE par lots — reinforce.py

Leçon 6 : le réseau apprend à jouer en essayant, avec la règle
    perte = − log π(action) × avantage
où l'avantage dit si le coup a fait mieux ou moins bien que la moyenne.

Trois temps par itération :
    1. jouer B parties avec le réseau, sans gradient (rapide, peu de mémoire) ;
    2. calculer pour chaque coup le retour : récompenses futures cumulées (facteur GAMMA) ;
    3. rejouer les mêmes décisions AVEC gradient, par petits paquets, puis un pas d'Adam.
"""
import numpy as np
import torch
import torch.nn.functional as F

from env_lots import SudokuEnvLots, refus_en_canaux
from reseau import masques_actions

GAMMA = 0.99


def grilles_presque_finies(puzzles, solutions, n_vides, rng):
    """Remplit avec la solution toutes les cases vides sauf n_vides (tirées au hasard) par grille."""
    sortie = puzzles.copy()
    for i in range(len(sortie)):
        vides = np.flatnonzero(sortie[i].ravel() == 0)
        a_remplir = rng.permutation(vides)[n_vides:]
        sortie[i].ravel()[a_remplir] = solutions[i].ravel()[a_remplir]
    return sortie


@torch.no_grad()
def jouer_lot(env, modele, horizon, generateur=None, glouton=False):
    """Joue jusqu'à `horizon` coups. Retourne grilles (T,B,9,9), actions, récompenses, actifs (T,B)
    et refus (T,B,729) : les coups refusés vus par le réseau à chaque tour (None s'il ne les voit pas)."""
    grilles, actions, recompenses, actifs, refus = [], [], [], [], []
    for _ in range(horizon):
        if bool(env.termine.all()):
            break
        masques = env.masques()
        logits, _ = modele(env.etats(modele.voit_refus), masques)
        if glouton:
            a = logits.argmax(dim=1)
        else:
            a = torch.multinomial(F.softmax(logits, dim=1), 1, generator=generateur).squeeze(1)
        actifs.append(~env.termine)
        grilles.append(env.grille.to(torch.int8))
        if modele.voit_refus:
            refus.append(env.refuse.clone())
        _, r, _, _ = env.step(a)
        actions.append(a)
        recompenses.append(r)
    return (torch.stack(grilles), torch.stack(actions), torch.stack(recompenses),
            torch.stack(actifs), torch.stack(refus) if refus else None)


def retours(recompenses, gamma=GAMMA):
    """G_t = r_t + gamma × G_{t+1}, calculé de la fin vers le début. (T,B) → (T,B)."""
    G = torch.zeros_like(recompenses)
    suite = torch.zeros_like(recompenses[0])
    for t in range(len(recompenses) - 1, -1, -1):
        suite = recompenses[t] + gamma * suite
        G[t] = suite
    return G


def pas_reinforce(modele, optimiseur, grilles, actions, G, actifs, refus=None, paquet=1024, bonus_entropie=0.0):
    """Un pas d'apprentissage. Retourne (perte, entropie moyenne) sur les coups réellement joués.

    refus (T,B,729) : coups refusés vus à chaque tour, requis si le réseau les voit.
    bonus_entropie : poids du bonus d'entropie (0 = aucun) ; il empêche la politique de devenir
    déterministe trop tôt (un réseau à entropie 0 n'explore plus et n'apprend plus)."""
    g = grilles[actifs].long()
    a = actions[actifs]
    r = refus[actifs] if refus is not None else None
    # Base de comparaison PAR NUMÉRO DE COUP : le retour dépend surtout de la place dans la partie
    # (début ≈ 27, fin ≈ 10), pas de la qualité du coup. On compare chaque coup à ceux joués au même tour.
    base = (G * actifs).sum(1) / actifs.sum(1).clamp(min=1)
    ret = (G - base[:, None])[actifs]
    avantage = ret / (ret.std() + 1e-8)
    N = len(a)
    ordre = torch.randperm(N, device=a.device)
    optimiseur.zero_grad()
    perte_totale, entropie_totale = 0.0, 0.0
    for debut in range(0, N, paquet):
        idx = ordre[debut:debut + paquet]
        etats = F.one_hot(g[idx], 10).permute(0, 3, 1, 2).float()
        if r is not None:
            etats = torch.cat([etats, refus_en_canaux(r[idx])], dim=1)
        logits, _ = modele(etats, masques_actions(g[idx]))
        logp = F.log_softmax(logits, dim=1)
        logp_a = logp.gather(1, a[idx, None]).squeeze(1)
        perte = -(logp_a * avantage[idx]).sum() / N
        if bonus_entropie:
            # bonus : on soustrait bonus × entropie à la perte, donc on récompense l'hésitation
            entropie = -(logp.exp() * logp.clamp(min=-30)).sum(dim=1)
            perte = perte - bonus_entropie * entropie.sum() / N
        perte.backward()
        with torch.no_grad():
            p = logp.exp()
            entropie_totale += float(-(p * logp.clamp(min=-30)).sum(dim=1).sum())
            perte_totale += float(perte)
    optimiseur.step()
    return perte_totale, entropie_totale / N


@torch.no_grad()
def evaluer_lot(modele, puzzles, solutions, n_coups_max, appareil, glouton=False, graine=0, memoire=False):
    """Joue les grilles données jusqu'à la fin ou au timeout. Retourne un dictionnaire de mesures."""
    modele.eval()
    env = SudokuEnvLots(puzzles, solutions, n_coups_max=n_coups_max, appareil=appareil, memoire=memoire)
    gen = torch.Generator(device=appareil).manual_seed(graine)
    retour = torch.zeros(env.B, device=appareil)
    for _ in range(n_coups_max):
        if bool(env.termine.all()):
            break
        logits, _ = modele(env.etats(modele.voit_refus), env.masques())
        a = logits.argmax(1) if glouton else torch.multinomial(F.softmax(logits, 1), 1, generator=gen).squeeze(1)
        _, r, _, _ = env.step(a)
        retour += r
    modele.train()
    resolues = env.n_vides == 0
    return {"resolues": float(resolues.float().mean()),
            "retour": float(retour.mean()),
            "coups": float(env.n_coups.float().mean())}
