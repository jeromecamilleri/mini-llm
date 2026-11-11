"""Lecture d'un fichier Kindle (.mobi / .azw3) écrite à la main.

Structure d'un fichier MOBI (format hérité de Palm) :

    en-tête PalmDB  ->  liste de « records » (blocs) avec leur position
    record 0        ->  en-tête PalmDOC + en-tête MOBI (compression, encodage...)
    records 1..N    ->  le texte HTML, découpé en blocs de 4096 octets, compressés
    records suivants -> images, index... (ignorés ici)

La compression PalmDOC est un LZ77 : on recopie soit des octets littéraux,
soit un morceau déjà écrit (distance + longueur).
"""

import html
import re
import struct


def lire_records(donnees):
    """Renvoie la liste des records (blocs d'octets) du fichier PalmDB."""
    nb = struct.unpack(">H", donnees[76:78])[0]
    debuts = [struct.unpack(">I", donnees[78 + 8 * i : 82 + 8 * i])[0] for i in range(nb)]
    fins = debuts[1:] + [len(donnees)]
    return [donnees[d:f] for d, f in zip(debuts, fins)]


def decompresser_palmdoc(bloc):
    """Décompression LZ77 « PalmDOC » d'un bloc de texte."""
    sortie = bytearray()
    i = 0
    while i < len(bloc):
        c = bloc[i]
        i += 1
        if c == 0 or 0x09 <= c <= 0x7F:
            sortie.append(c)  # octet littéral
        elif 0x01 <= c <= 0x08:
            sortie += bloc[i : i + c]  # c octets littéraux
            i += c
        elif c >= 0xC0:
            sortie.append(0x20)  # une espace...
            sortie.append(c ^ 0x80)  # ...suivie d'un caractère
        else:  # 0x80..0xBF : copie d'un morceau déjà écrit
            paire = ((c << 8) | bloc[i]) & 0x3FFF
            i += 1
            distance, longueur = paire >> 3, (paire & 7) + 3
            for _ in range(longueur):
                sortie.append(sortie[-distance])
    return bytes(sortie)


def _taille_fin(bloc, drapeaux):
    """Nombre d'octets « parasites » ajoutés à la fin d'un bloc compressé."""
    total = 0
    for bit in range(15, 0, -1):
        if drapeaux >> bit & 1:
            valeur, decalage = 0, 0
            for octet in reversed(bloc[: len(bloc) - total]):
                valeur |= (octet & 0x7F) << decalage
                decalage += 7
                if octet & 0x80:
                    break
            total += valeur
    if drapeaux & 1:
        total += (bloc[len(bloc) - total - 1] & 3) + 1
    return total


def extraire_html(chemin):
    """Renvoie le texte HTML brut d'un livre et un dictionnaire d'infos."""
    with open(chemin, "rb") as f:
        records = lire_records(f.read())
    r0 = records[0]
    compression, _, longueur, nb_blocs, _, chiffrement = struct.unpack(">HHIHHH", r0[:14])
    if chiffrement != 0:
        raise ValueError("livre chiffré (DRM)")
    if compression != 2:
        raise ValueError(f"compression {compression} non gérée")
    encodage = struct.unpack(">I", r0[28:32])[0]
    drapeaux = 0
    if struct.unpack(">I", r0[20:24])[0] >= 0xE4:
        drapeaux = struct.unpack(">H", r0[0xF2:0xF4])[0]
    texte = b""
    for bloc in records[1 : nb_blocs + 1]:
        texte += decompresser_palmdoc(bloc[: len(bloc) - _taille_fin(bloc, drapeaux)])
    codec = "utf-8" if encodage == 65001 else "cp1252"
    infos = {"encodage": codec, "octets": len(texte), "blocs": nb_blocs}
    return texte.decode(codec, errors="replace"), infos


def html_vers_texte(contenu):
    """Retire les balises HTML (version simple : le vrai nettoyage vient après)."""
    contenu = re.sub(r"(?is)<(script|style).*?</\1>", "", contenu)
    contenu = re.sub(r"(?i)</?(p|div|br|h[1-6]|li|tr|blockquote)[^>]*>", "\n", contenu)
    contenu = re.sub(r"<[^>]+>", "", contenu)
    contenu = html.unescape(contenu)
    contenu = re.sub(r"[ \t\u00a0]+", " ", contenu)
    contenu = re.sub(r"\n\s*\n+", "\n\n", contenu)
    return contenu.strip()


def lire_metadonnees(chemin):
    """Titre, auteur, langue... lus dans le bloc EXTH de l'en-tête MOBI."""
    with open(chemin, "rb") as f:
        r0 = lire_records(f.read())[0]
    debut = r0.find(b"EXTH")
    if debut < 0:
        return {}
    nb = struct.unpack(">I", r0[debut + 8 : debut + 12])[0]
    types = {100: "auteur", 503: "titre", 524: "langue", 101: "editeur"}
    infos, i = {}, debut + 12
    for _ in range(nb):
        type_, taille = struct.unpack(">II", r0[i : i + 8])
        if type_ in types:
            infos[types[type_]] = r0[i + 8 : i + taille].decode("utf-8", errors="replace")
        i += taille
    return infos
