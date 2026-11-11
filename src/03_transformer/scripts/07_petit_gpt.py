"""
LEÇON 7 (transformer) : le petit GPT complet
============================================

    .venv/bin/python src/03_transformer/scripts/07_petit_gpt.py

PLACE DANS LE PARCOURS (chapitre 03_transformer)
C'est la dernière leçon du chapitre : on assemble TOUT ce qu'on a vu.

L'ESSENTIEL
Un GPT = une machine qui prend des mots et rend, pour CHAQUE position, un score
pour chaque mot du vocabulaire : « quel mot vient après ? ». Le tuyau est :

  mots -> identifiants -> vecteurs (+ positions) -> bloc 1 -> ... -> bloc N -> scores par mot

Ici les poids sont tirés AU HASARD : le modèle n'a rien appris, il génère du
charabia. C'est normal ! Le chapitre 04_training servira à l'entraîner.

LE SIGLE GPT
  G = Generative   : il GÉNÈRE du texte, un mot après l'autre (05_inference).
  P = Pre-trained  : il est d'abord PRÉ-ENTRAÎNÉ à prédire le mot suivant sur un
                     énorme corpus, avant d'être adapté (04_training).
  T = Transformer  : l'ARCHITECTURE : attention + feed forward, empilés (ce chapitre).
Le « Chat » de ChatGPT désigne l'interface de conversation, pas l'architecture.

GLOUTON, TIRAGE, TEMPÉRATURE (comment choisir le mot suivant)
  glouton : on prend toujours le mot le plus probable (sûr, mais répétitif).
  tirage  : on tire au sort selon les probabilités (varié, mais peut dérailler).
  température : réglage du tirage, softmax(scores / T). Basse (0,5) : le favori
                domine (proche du glouton). 1 : probabilités du modèle. Haute (2) :
                probabilités aplaties, plus de hasard.
  Exemple, scores 2,0 / 1,5 / 1,0 : T=0,5 -> 67/24/9 % ; T=1 -> 51/31/19 % ; T=2 -> 42/33/25 %.

LES FORMES (B = 1 ici : une seule phrase à la fois)
  identifiants      (T,)
  embeddings + pos. (T, d)            <- chaque mot devient d nombres
  après chaque bloc (T, d)            <- même forme : on peut empiler
  scores par mot    (T, vocabulaire)  <- d nombres deviennent « un score par mot »
Les poids seuls changent la taille : le nombre de paramètres.

PREMIER REPÈRE : la perte d'un modèle au hasard
Sans rien savoir, le modèle donne à peu près la même probabilité à chaque mot :
1 / V pour un vocabulaire de V mots, donc une perte proche de ln(V). Après
entraînement, la perte doit descendre bien en dessous : c'est le signal que
quelque chose a été appris.
"""
import sys
from pathlib import Path

import numpy as np

racine = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(racine / "tools"))
sys.path.insert(0, str(racine.parent / "02_tokenizer" / "tools"))
from gpt import generer, gpt, init_gpt, nb_parametres, perte
from tokenizer_basic import TokenizerMots

corpus = (racine.parent / "02_tokenizer" / "corpus_animaux.txt").read_text(encoding="utf-8")
tok = TokenizerMots(corpus)
V = len(tok.vocab.token_vers_id)

# Taille modeste : on veut VOIR le mécanisme. Le vrai mini-GPT sera plus grand.
d, n_tetes, n_blocs, T_max = 8, 2, 2, 8
p = init_gpt(V, d, n_tetes, n_blocs, T_max, np.random.default_rng(0))

print("ÉTAPE 1 : la taille du modèle")
print(f"  vocabulaire V = {V} mots, d = {d}, {n_tetes} têtes, {n_blocs} blocs, T_max = {T_max}")
print(f"  paramètres : {nb_parametres(p)}")

print("\nÉTAPE 2 : le chemin d'une phrase (formes)")
ids = tok.encoder("le chat dort .")
print(f"  phrase {tok.decoder(ids)!r} -> identifiants {ids}")
logits, formes = gpt(ids, p, trace=True)
for nom, forme in formes:
    print(f"  {nom:<24} {forme}")

print("\nÉTAPE 3 : les scores de la dernière position")
print("  La dernière ligne répond à : « quel mot après `.` ? »")
top = np.argsort(logits[-1])[::-1][:3]
for i in top:
    print(f"  {tok.vocab.decoder_ids([int(i)])[0]:<8} score {logits[-1][i]:+.2f}")
print("  (au hasard : aucune raison que ce soit juste)")

print("\nÉTAPE 4 : la perte d'un modèle qui ne sait rien")
phrase = tok.encoder("le chat dort . le chien mange .")
print(f"  perte mesurée sur une phrase du corpus : {perte(phrase, p):.3f}")
print(f"  ln(V) = ln({V}) = {np.log(V):.3f}  <- l'ordre de grandeur attendu")

print("\nÉTAPE 5 : générer du texte (glouton, puis tirage au hasard)")
debut = tok.encoder("le chat")
print("  glouton :", tok.decoder(generer(debut, p, 8, T_max)))
print("  tirage  :", tok.decoder(generer(debut, p, 8, T_max, rng=np.random.default_rng(1))))
print("  -> du charabia, comme prévu. Le chapitre 04_training va changer ça.")

print("\nÉTAPE 6 : le début compte seul, pas le futur (rappel leçon 2)")
a = gpt(tok.encoder("le chat dort ."), p)
b = gpt(tok.encoder("le chat dort le"), p)
print("  on change le DERNIER mot : les 3 premières lignes de scores restent-elles identiques ?",
      np.allclose(a[:3], b[:3]))
