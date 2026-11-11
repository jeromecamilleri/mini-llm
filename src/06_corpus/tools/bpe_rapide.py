"""BPE rapide : même idée que 02_tokenizer/tools/bpe.py, prévu pour des millions de mots.

Deux accélérations (voir scripts/05_bpe_corpus.py) :
  - on garde en mémoire le décompte de toutes les paires et on ne le met à jour que pour
    les mots touchés par la fusion (au lieu de tout recompter à chaque fusion) ;
  - à l'encodage, on mémorise le découpage de chaque mot déjà rencontré.
"""

import heapq
import json
import re
from collections import Counter, defaultdict

DECOUPAGE = re.compile(r" ?\w+|[^\w\s]|\s")  # un mot garde l'espace qui le précède


def compter_morceaux(texte, compteur=None):
    """Compte combien de fois chaque morceau (mot avec son espace, ponctuation) apparaît."""
    compteur = Counter() if compteur is None else compteur
    compteur.update(DECOUPAGE.findall(texte))
    return compteur


def fusionner(tokens, paire):
    """Remplace chaque occurrence de la paire (a, b) par le token « ab »."""
    a, b = paire
    resultat, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            resultat.append(a + b)
            i += 2
        else:
            resultat.append(tokens[i])
            i += 1
    return resultat


def entrainer(morceaux, n_fusions, rapport=None):
    """morceaux : {morceau: fréquence}. Renvoie la liste ordonnée des paires fusionnées."""
    mots = [list(m) for m in morceaux]
    freqs = list(morceaux.values())
    compte = Counter()
    ou = defaultdict(set)  # paire -> indices des mots qui (ont pu) la contenir
    for i, (mot, f) in enumerate(zip(mots, freqs)):
        for paire in zip(mot, mot[1:]):
            compte[paire] += f
            ou[paire].add(i)
    tas = [(-n, paire) for paire, n in compte.items()]
    heapq.heapify(tas)  # le plus fréquent en haut ; les entrées périmées sont ignorées au passage

    fusions = []
    while len(fusions) < n_fusions and tas:
        n, paire = heapq.heappop(tas)
        if compte[paire] != -n or -n <= 0:
            continue  # entrée périmée : le compte a changé depuis
        fusions.append(paire)
        modifiees = set()
        for i in ou.pop(paire):
            ancien = mots[i]
            nouveau = fusionner(ancien, paire)
            if len(nouveau) == len(ancien):
                continue
            f = freqs[i]
            for p in zip(ancien, ancien[1:]):
                compte[p] -= f
                modifiees.add(p)
            for p in zip(nouveau, nouveau[1:]):
                compte[p] += f
                ou[p].add(i)
                modifiees.add(p)
            mots[i] = nouveau
        for p in modifiees:
            if compte[p] > 0:
                heapq.heappush(tas, (-compte[p], p))
        if rapport and len(fusions) % rapport == 0:
            print(f"    {len(fusions):>6} fusions : {paire[0]!r} + {paire[1]!r} (vue {-n:,} fois)")
    return fusions


class BPE:
    def __init__(self, caracteres, fusions):
        self.fusions = [tuple(p) for p in fusions]
        self.rangs = {p: r for r, p in enumerate(self.fusions)}
        tokens = ["<inc>"] + sorted(caracteres) + [a + b for a, b in self.fusions]
        self.id_de = {t: i for i, t in enumerate(tokens)}
        self.tokens = tokens
        self.cache = {}

    @property
    def taille(self):
        return len(self.tokens)

    def decouper_morceau(self, morceau):
        """Applique les fusions dans l'ordre où elles ont été apprises (la plus ancienne d'abord)."""
        if morceau not in self.cache:
            tokens = list(morceau)
            while len(tokens) > 1:
                rang, paire = min((self.rangs.get(p, 1 << 60), p) for p in zip(tokens, tokens[1:]))
                if rang == 1 << 60:
                    break
                tokens = fusionner(tokens, paire)
            self.cache[morceau] = tokens
        return self.cache[morceau]

    def encoder(self, texte):
        ids = []
        for morceau in DECOUPAGE.findall(texte):
            ids.extend(self.id_de.get(t, 0) for t in self.decouper_morceau(morceau))
        return ids

    def decoder(self, ids):
        return "".join(self.tokens[i] for i in ids)

    def sauver(self, chemin):
        caracteres = [t for t in self.tokens[1 : len(self.tokens) - len(self.fusions)]]
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump({"caracteres": caracteres, "fusions": self.fusions}, f, ensure_ascii=False)

    @classmethod
    def charger(cls, chemin):
        with open(chemin, encoding="utf-8") as f:
            donnees = json.load(f)
        return cls(donnees["caracteres"], donnees["fusions"])
