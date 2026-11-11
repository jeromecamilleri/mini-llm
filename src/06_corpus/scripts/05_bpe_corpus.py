"""06_corpus — Leçon 5 : un tokenizer BPE entraîné sur les livres.

Un tokenizer découpe le texte en « tokens » (morceaux de mots). Le BPE de
02_tokenizer (leçon 3) marchait sur 67 mots. Ici : 36 millions de mots.
Deux problèmes, deux solutions (tools/bpe_rapide.py) :

  1. Compter les paires de tokens, à chaque fusion, sur tout le corpus : trop lent.
     -> Un mot a toujours le même découpage : on compte chaque mot DIFFÉRENT
        une seule fois, avec sa fréquence (« ␣de » apparaît 1 000 000 de fois,
        on ne le traite qu'une fois). Puis, après une fusion, on ne met à jour
        que les mots qui contiennent la paire fusionnée.
  2. Encoder : rejouer des milliers de fusions sur chaque mot : trop lent.
     -> On cherche à chaque fois la fusion la plus ancienne possible dans le
        mot, et on mémorise le résultat de chaque mot déjà vu.

Le résultat (les fusions) est écrit dans datasets/tokenized/bpe_NNNN.json.

Réglage : N_FUSIONS. Vocabulaire final = 1 (<inc>) + caractères (~110) + fusions.
Un vocabulaire plus grand = des séquences plus courtes, mais une table
d'embeddings plus grosse. Pour la GTX 1070, quelques milliers suffisent.
"""

import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bpe_rapide import BPE, compter_morceaux, entrainer  # noqa: E402

NETTOYE = RACINE / "datasets" / "cleaned"
EXCLUSIONS = RACINE / "datasets" / "exclusions.txt"
SORTIE = RACINE / "datasets" / "tokenized"
N_FUSIONS = 4000
MAX_LIVRES = None  # un entier pour tester vite sur quelques livres

SORTIE.mkdir(parents=True, exist_ok=True)
exclus = set(EXCLUSIONS.read_text(encoding="utf-8").splitlines()) if EXCLUSIONS.exists() else set()
fichiers = [f for f in sorted(NETTOYE.glob("*.txt")) if f.name not in exclus][:MAX_LIVRES]

print(f"ÉTAPE 1 — compter les mots ({len(fichiers)} livres, {len(exclus)} exclus)")
debut = time.time()
morceaux = compter_morceaux("")
for f in fichiers:
    compter_morceaux(f.read_text(encoding="utf-8"), morceaux)
total_morceaux = sum(morceaux.values())
print(f"  {total_morceaux:,} morceaux, dont {len(morceaux):,} différents ({time.time() - debut:.0f} s)")
print("  les plus fréquents :", morceaux.most_common(8))
print("  une seule occurrence :", f"{sum(1 for n in morceaux.values() if n == 1):,} morceaux")
# Beaucoup de morceaux n'apparaissent qu'une fois (noms propres, fautes d'OCR) :
# c'est pour eux que les sous-mots sont utiles.

print(f"\nÉTAPE 2 — {N_FUSIONS} fusions")
debut = time.time()
fusions = entrainer(morceaux, N_FUSIONS, rapport=500)
print(f"  terminé en {time.time() - debut:.0f} s")
caracteres = sorted({c for m in morceaux for c in m})
bpe = BPE(caracteres, fusions)
print(f"  vocabulaire : {bpe.taille:,} tokens ({len(caracteres)} caractères + {len(fusions)} fusions)")
print("  10 premières fusions :", [a + b for a, b in fusions[:10]])
print("  10 dernières :", [a + b for a, b in fusions[-10:]])

print("\nÉTAPE 3 — ce que ça donne")
total_caracteres = sum(len(m) * n for m, n in morceaux.items())
total_tokens = sum(len(bpe.decouper_morceau(m)) * n for m, n in morceaux.items())
print(f"  {total_caracteres:,} caractères -> {total_tokens:,} tokens "
      f"({total_caracteres / total_tokens:.2f} caractères par token)")
for phrase in ["Le chat dort sur le canapé.", "L'intelligence artificielle", "Zorglubissime"]:
    ids = bpe.encoder(phrase)
    print(f"  {phrase!r} -> {[bpe.tokens[i] for i in ids]}")
    assert bpe.decoder(ids) == phrase  # on retrouve exactement le texte

chemin = SORTIE / f"bpe_{N_FUSIONS}.json"
bpe.sauver(chemin)
print(f"\nécrit : {chemin.relative_to(RACINE)}")
