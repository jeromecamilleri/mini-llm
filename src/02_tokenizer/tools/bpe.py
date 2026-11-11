"""BPE (Byte Pair Encoding) : sous-mots obtenus en fusionnant les paires les plus fréquentes.

Détail pas à pas dans scripts/03_bpe.py. Le texte est découpé en « morceaux » (un mot avec
l'espace qui le précède, ou un signe de ponctuation) ; les fusions ne traversent jamais
un morceau. Ainsi décoder = recoller les tokens, sans rien perdre.
"""
import re
from collections import Counter

from vocabulary import Vocabulaire


def decouper_morceaux(texte):
    """' le chat.' -> [' le', ' chat', '.'] : un mot garde l'espace qui le précède."""
    return re.findall(r" ?\w+|[^\w\s]|\s", texte)


def compter_paires(morceaux):
    """morceaux : {tuple de tokens: fréquence}. Retourne le nombre de fois où chaque paire apparaît."""
    paires = Counter()
    for tokens, frequence in morceaux.items():
        for paire in zip(tokens, tokens[1:]):
            paires[paire] += frequence
    return paires


def fusionner(tokens, paire):
    """Remplace chaque occurrence de la paire (a, b) par le token « ab »."""
    resultat, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == paire:
            resultat.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            resultat.append(tokens[i])
            i += 1
    return tuple(resultat)


class BPE:
    def __init__(self, corpus, n_fusions):
        morceaux = Counter(tuple(m) for m in decouper_morceaux(corpus))
        self.fusions = []  # liste ordonnée de paires fusionnées
        self.historique = []  # (paire, fréquence) de chaque fusion
        for _ in range(n_fusions):
            paires = compter_paires(morceaux)
            if not paires:
                break
            paire, frequence = paires.most_common(1)[0]  # ex æquo : la première rencontrée
            self.fusions.append(paire)
            self.historique.append((paire, frequence))
            morceaux = self._appliquer(morceaux, paire)
        caracteres = sorted({c for m in decouper_morceaux(corpus) for c in m})
        self.vocab = Vocabulaire(caracteres + [a + b for a, b in self.fusions])

    @staticmethod
    def _appliquer(morceaux, paire):
        nouveaux = Counter()
        for tokens, frequence in morceaux.items():
            nouveaux[fusionner(tokens, paire)] += frequence
        return nouveaux

    def decouper(self, texte):
        """Texte -> liste de tokens (en appliquant les fusions dans l'ordre où elles ont été apprises)."""
        resultat = []
        for morceau in decouper_morceaux(texte):
            tokens = tuple(morceau)
            for paire in self.fusions:
                tokens = fusionner(tokens, paire)
            resultat.extend(tokens)
        return resultat

    def encoder(self, texte):
        return self.vocab.encoder_unites(self.decouper(texte))

    def decoder(self, ids):
        return "".join(self.vocab.decoder_ids(ids))
