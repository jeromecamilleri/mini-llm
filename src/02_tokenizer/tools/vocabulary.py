"""Vocabulaire : table de correspondance entre unités de texte (tokens) et identifiants.

Détail pas à pas dans scripts/01_caracteres.py. Ici une unité = un token quelconque
(un caractère pour l'instant, des mots ou des sous-mots, morceaux de mots, plus tard).
"""


class Vocabulaire:
    def __init__(self, tokens):
        """tokens : liste des unités distinctes ; l'identifiant est la position dans la liste."""
        self.id_vers_token = list(tokens)
        self.token_vers_id = {t: i for i, t in enumerate(self.id_vers_token)}

    @classmethod
    def depuis_unites(cls, unites):
        """Construit le vocabulaire à partir d'un texte découpé : unités distinctes, triées."""
        return cls(sorted(set(unites)))

    def __len__(self):
        return len(self.id_vers_token)

    def encoder_unites(self, unites):
        """Unités -> identifiants. Lève KeyError si une unité est absente du vocabulaire."""
        return [self.token_vers_id[u] for u in unites]

    def decoder_ids(self, ids):
        """Identifiants -> liste d'unités."""
        return [self.id_vers_token[i] for i in ids]
