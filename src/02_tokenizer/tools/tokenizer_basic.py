"""Tokenizer de mots avec un jeton « inconnu ».

Détail pas à pas dans scripts/02_mots.py. Réutilise Vocabulaire (vocabulary.py).
"""
import re

from vocabulary import Vocabulaire

INCONNU = "<inc>"


def decouper_mots(texte):
    """Minuscules ; un mot = suite de lettres/chiffres ; chaque ponctuation est un token."""
    return re.findall(r"\w+|[^\w\s]", texte.lower())


class TokenizerMots:
    def __init__(self, corpus):
        mots = decouper_mots(corpus)
        # <inc> a l'identifiant 0 : il remplace tout mot absent du corpus.
        self.vocab = Vocabulaire([INCONNU] + sorted(set(mots)))

    def encoder(self, texte):
        ids = self.vocab.token_vers_id
        return [ids.get(m, ids[INCONNU]) for m in decouper_mots(texte)]

    def decoder(self, ids):
        return " ".join(self.vocab.decoder_ids(ids))
