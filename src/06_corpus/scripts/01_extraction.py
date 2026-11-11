"""06_corpus — Leçon 1 : extraire le texte d'un livre Kindle.

Un livre Kindle n'est pas du texte : c'est un fichier binaire. Un modèle
de langage a besoin de texte brut. Il faut donc ouvrir le fichier.

Un fichier .mobi contient le livre en HTML, découpé en blocs et compressé.
On fait trois choses (voir tools/mobi.py) :
  1. lire la liste des blocs (« records ») ;
  2. décompresser chaque bloc (LZ77 « PalmDOC ») ;
  3. retirer les balises HTML.

On teste sur 3 livres. Le texte est écrit dans datasets/extracted/.
Les originaux ne sont jamais modifiés.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mobi import extraire_html, html_vers_texte  # noqa: E402

ORIGINAUX = RACINE / "datasets" / "originals"
SORTIE = RACINE / "datasets" / "extracted"

TOUS_LES_LIVRES = True  # True : traite les 353 livres de datasets/originals/

LIVRES = [
    "Voltaire-L_Ingenu_QWZ5LG7YDJTF6Y3QA4CMDGBKP4TP3ZEM.azw3",
    "Poids Du Papillon, Le - Erri De Luca.mobi",
    "Faute d'egalite - Bergounioux, Pierre.mobi",
]
if TOUS_LES_LIVRES:
    LIVRES = sorted(p.name for p in ORIGINAUX.iterdir() if p.suffix in (".mobi", ".azw3"))

SORTIE.mkdir(parents=True, exist_ok=True)

print(f"ÉTAPE 1 — extraction de {len(LIVRES)} livres")
echecs = []
for nom in LIVRES:
    try:
        contenu, infos = extraire_html(ORIGINAUX / nom)
    except Exception as erreur:  # un livre cassé ne doit pas arrêter les autres
        echecs.append((nom, erreur))
        continue
    texte = html_vers_texte(contenu)
    (SORTIE / (Path(nom).stem + ".txt")).write_text(texte, encoding="utf-8")
    if not TOUS_LES_LIVRES:
        print(f"\n{nom}")
        print(f"  HTML : {infos['octets']:,} octets ({infos['blocs']} blocs, {infos['encodage']})")
        print(f"  texte : {len(texte):,} caractères, {len(texte.split()):,} mots")
        print(f"  début : {texte[:400]!r}")
        milieu = len(texte) // 2
        print(f"  milieu : {texte[milieu:milieu + 300]!r}")

print(f"\n{len(LIVRES) - len(echecs)} livres extraits, {len(echecs)} échecs")
for nom, erreur in echecs:
    print(f"  {nom} : {erreur}")
