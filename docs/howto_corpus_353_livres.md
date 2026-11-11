# How-to : un GPT entraîné sur mes 353 livres Kindle

Pas à pas, de la Kindle au modèle. Chaque étape dit ce qu'elle fait, la commande, et ce qu'on obtient. Les étapes 1 à 8 sont faites, la 9 est écrite (à lancer), la 10 reste à écrire.

Les livres sont protégés par le droit d'auteur : tout reste en local, rien n'est commité (`.gitignore`), usage personnel seulement.

## Vue d'ensemble

```
Kindle ─1─> datasets/originals ─2─> extracted ─3─> cleaned ─4─> metadata.csv ─5─> triage
                                                                         │
                       modèle <─9─ entraînement <─8─ tokenized <─7─ BPE <┘ (6 : exclusions)
```

| # | Étape | Dossier / script | État |
|---|---|---|---|
| 1 | Copier les livres | à la main | fait |
| 2 | Extraire le texte | `src/06_corpus/scripts/01_extraction.py` | fait |
| 3 | Nettoyer | `02_nettoyage.py` | fait |
| 4 | Métadonnées et statistiques | `03_metadonnees.py` | fait |
| 5 | Repérer les livres suspects | `04_triage.py` | fait |
| 6 | Écarter les mauvais livres | `datasets/exclusions.txt` | fait (7 écartés) |
| 7 | Entraîner un tokenizer BPE sur le corpus | `05_bpe_corpus.py` | fait |
| 8 | Découper le corpus en tokens | `06_tokenisation.py` | fait |
| 9 | Entraîner un GPT sur les livres | `07_entrainement_corpus.py` | écrit, à lancer |
| 10 | Évaluer et générer | à écrire | à faire |

## Prérequis

- Projet installé avec son venv : `.venv/bin/python` (PyTorch pour l'étape 9, GTX 1070 de 8 Go).
- Les chapitres `02_tokenizer` à `05_inference` servent de base : on réutilise leurs outils (BPE, GPT PyTorch, boucle d'entraînement, checkpoints, génération).
- Si le premier lancement de `.venv/bin/python` répond « No module named torch », relance la commande.

## 1. Copier les livres de la Kindle

La Kindle est montée dans `/run/media/camillej/Kindle/documents`. On copie dans `datasets/originals/` :

- tous les `.mobi` (193 fichiers, sans doublon de nom) ;
- tous les `.azw3` (160) sauf les extraits `_sample` et le guide Kindle.

Les `.azw3` sont des fichiers MOBI plus récents (en-tête `BOOKMOBI`, aucun DRM), pas des doublons des `.mobi`. Non copiés : `.azw` (36, en-tête illisible), `.kfx`, `.pdf`, `.epub`.

Total : **353 livres, 570 Mo**. On ne modifie jamais `datasets/originals/`.

## 2. Extraire le texte (leçon 1)

Un `.mobi` / `.azw3` est du HTML, découpé en blocs de 4096 octets et compressé en LZ77 « PalmDOC ». `tools/mobi.py` lit les blocs, les décompresse et retire les balises.

Dans `src/06_corpus/scripts/01_extraction.py`, mets `TOUS_LES_LIVRES = True`, puis :

```bash
.venv/bin/python src/06_corpus/scripts/01_extraction.py
```

Résultat : un `.txt` par livre dans `datasets/extracted/`. Un livre qui échoue est signalé à la fin sans arrêter les autres. Environ 0,1 s par livre ; le GPU ne sert à rien ici, c'est du traitement d'octets.

## 3. Nettoyer (leçon 2)

```bash
.venv/bin/python src/06_corpus/scripts/02_nettoyage.py
```

Quatre règles (`tools/nettoyage.py`) :

| Règle | Effet |
|---|---|
| `retirer_gutenberg` | retire l'en-tête et la licence anglaise du Projet Gutenberg |
| `normaliser` | `’ “ ” — …` et espace insécable deviennent `' " - ...` et espace |
| `reparer_lettrines` | `Àl'heure` devient `À l'heure` |
| `refaire_paragraphes` | une ligne = un paragraphe ; retire les numéros de page |

Résultat : `datasets/cleaned/`. Ce qui reste : pages de titre et de garde, mentions d'éditeur, erreurs d'OCR.

## 4. Métadonnées (leçon 3)

```bash
.venv/bin/python src/06_corpus/scripts/03_metadonnees.py
```

Écrit `datasets/metadata.csv`, une ligne par livre :

- lus dans le fichier Kindle (bloc EXTH) : `titre`, `auteur`, `editeur`, `langue_declaree` ;
- mesurés sur le texte nettoyé : `caracteres`, `mots`, `mots_differents`, `paragraphes`, `lettres_par_mot` ;
- calculée : `langue_estimee` ; et `fichier` pour retrouver le texte.

**Corpus actuel : 36,9 millions de mots, 223 millions de caractères**, médiane de 93 000 mots par livre.

## 5. Repérer les suspects (leçon 4)

```bash
.venv/bin/python src/06_corpus/scripts/04_triage.py
```

Critères : moins de 5 000 mots, langue estimée autre que le français, langue déclarée fausse, doublon (même titre et même auteur), diversité du vocabulaire anormale. Le script **propose** et écrit `datasets/a_verifier.csv` ; il ne supprime rien.

Résultat actuel : 27 suspects sur 353 :

- vrais problèmes : un livre à 8 mots (René Girard, extraction ratée), 3 textes anglais (*Animal Farm*, *Hernani*, *Phoenix Project*) ;
- 3 paires de doublons (Becky Chambers ×2, *Central Station*, *Les Voyageurs tome 3*) ;
- fausses alertes : une vingtaine de livres français dont l'étiquette Kindle est fausse ;
- à regarder à la main : *Anéantir* (mentions de mise en page) et le rapport « Lumières à l'ère numérique » (lettres espacées).

## 6. Écarter les mauvais livres

`datasets/exclusions.txt` : un nom de fichier par ligne (tel qu'affiché dans le triage). Les étapes suivantes ignorent ces fichiers. Fichier actuel : 7 livres écartés.

- extraction ratée : *René Girard, Des choses cachées* (8 mots) ;
- en anglais : *Animal Farm*, *Hernani*, *Phoenix Project* ;
- une seule copie gardée pour chaque doublon (Becky Chambers, *Central Station*, *Les Voyageurs tome 3*) ; les deux copies de chaque paire ont exactement le même nombre de mots.

Il reste **346 livres**. Les fausses étiquettes de langue sont gardées ; *Anéantir* et le rapport « Lumières à l'ère numérique » aussi (texte bruité mais bon).

## 7. Tokenizer BPE sur le corpus (leçon 5)

```bash
.venv/bin/python src/06_corpus/scripts/05_bpe_corpus.py
```

Le BPE de `02_tokenizer` marchait sur 67 mots ; sur 36 millions de mots il serait trop lent. `tools/bpe_rapide.py` accélère de deux façons :

- chaque mot **différent** n'est compté qu'une fois, avec sa fréquence ; après une fusion, on ne met à jour que les mots qui contiennent la paire ;
- à l'encodage, on applique la fusion la plus ancienne d'abord et on **mémorise** le découpage de chaque mot déjà vu.

Réglages dans le script : `N_FUSIONS = 4000` (vocabulaire final d'environ 4 100 tokens : `<inc>`, ~110 caractères, 4 000 fusions) et `MAX_LIVRES` (`None` = tous les livres ; un nombre, par exemple 40, pour un essai rapide).

Sortie : `datasets/tokenized/bpe_4000.json`, avec les fusions. Le script affiche le nombre de mots différents, les premières et dernières fusions, le nombre de caractères par token, et le découpage de quelques phrases (dont un mot inventé).

Vérifié sur un extrait de Voltaire : encoder puis décoder redonne exactement le texte. Comparé au BPE lent (60 000 caractères, 300 fusions) : 135 fusions identiques sur 300, les écarts venant des ex æquo ; nombre de tokens quasi identique (26 196 contre 26 199).

## 8. Tokeniser le corpus (leçon 6)

```bash
.venv/bin/python src/06_corpus/scripts/06_tokenisation.py
```

Chaque livre devient une suite d'ids (tokens). Deux fichiers sont écrits dans `datasets/tokenized/` : `train.bin` et `validation.bin` (ids sur 2 octets, `uint16`). La validation est faite de **livres entiers** (5 % des livres, tirage à graine fixe `GRAINE = 0`), jamais vus à l'entraînement : des morceaux tirés au hasard dans tous les livres paraîtraient trop faciles, le modèle ayant déjà vu les pages voisines. La liste est dans `validation_livres.txt`.

Résultat obtenu :

| | Livres | Tokens |
|---|---|---|
| `train.bin` | 329 | 66 216 164 (132 Mo) |
| `validation.bin` | 17 | 3 046 367 (6 Mo) |

- 0 token inconnu sur 69,3 millions ; le début décodé redonne le texte du livre.
- Durée : environ 40 s, tout sur le CPU (le GPU ne sert pas au traitement de texte).
- Livres de validation : *Hamlet* et *Pickwick II* en font partie, la validation est donc un peu inégale (théâtre, XIXe siècle). Changer la graine changerait les chiffres : ne pas le faire en cours d'expérience.
- Le tokenizer garde 611 caractères (contre 106 sur 3 livres) : beaucoup sont très rares. Amélioration possible plus tard, à tester seule.

## 9. Entraîner un GPT sur les livres (leçon 7)

```bash
.venv/bin/python src/06_corpus/scripts/07_entrainement_corpus.py
```

Même boucle que `04_training` (AdamW, validation régulière, meilleur checkpoint), à plus grande échelle :

| | Valeur |
|---|---|
| modèle | 4 blocs, d = 256, 4 têtes, T = 128 : **5 553 664 paramètres** |
| données | `train.bin` (66 M tokens), validation sur 17 livres entiers |
| batch | 32 fenêtres de 128 tokens (4 096 tokens par étape) |
| durée | `N_ETAPES = 20 000` (environ 1,2 passage sur le corpus) |
| taux | échauffement 300 étapes, 1e-3 puis cosinus jusqu'à 1e-4 ; gradient écrêté à 1 |

- Perte d'un modèle au hasard : ln(4612) = 8,44. Perplexité = exp(perte).
- Toutes les 500 étapes : perte train et validation, temps restant, texte généré à partir de « Il était une fois ».
- `MAX_MINUTES` fixe un budget de temps ; Ctrl-C sauve l'état ; `REPRENDRE = True` repart de `livres_dernier.pt`.
- Sorties : `models/checkpoints/livres_meilleur.pt`, `livres_dernier.pt`, `logs/entrainement_livres.csv`.
- Test de vérification du code (40 étapes) : validation 8,32 puis 7,51, texte encore en charabia, ce qui est normal au début.

## 10. Générer avec le modèle (leçon 8)

```
.venv/bin/python src/06_corpus/scripts/08_generation_corpus.py
```

- Recharge `livres_meilleur.pt` et le BPE de 4 000 fusions, sur le CPU (le GPU reste libre).
- Utilise les fonctions de `src/05_inference/tools/generation.py` : température, Top-K, Top-P. Même graine pour tous les réglages : seul le réglage change.
- Étape 2 : « Il était une fois » avec 5 réglages (T 0,3 / 0,8 / 1,2, Top-K 40, Top-P 0,9). Étape 3 : trois débuts différents.
- Lancée par Jérôme le 10 novembre 2026. Sortie complète : `logs/generation_livres_sortie.txt`.
- Courbe de perte : `09_courbe_perte.py`, lancée par Jérôme. Résultat : 40 mesures, validation 4,794 → 3,321, écart validation − train de 0,180 à la fin, gain par tranche de 500 étapes de 0,519 au début et 0,001 à la fin. Image : `logs/courbe_entrainement_livres.png`. Fiches `08` et `09` écrites, README à jour.

Ce que montrent les réglages (même graine, 120 tokens, début « Il était une fois ») :

| Réglage | Résultat |
|---|---|
| T 0,3 | texte correct mais **boucle** : « - Je ne sais pas. » répété 7 fois. Trop prudent, le favori gagne toujours. |
| T 0,8 | phrases variées et plausibles, sans fil (« Postrick », « Robin », « ma mère »). |
| T 1,2 | **déraille** : mots tronqués ou inventés (« survavant », « défav »). Trop de hasard. |
| T 0,8, Top-K 40 | un récit à la première personne, plus continu (le père, la chance), quelques absurdités. |
| T 0,8, Top-P 0,9 | proche de T 0,8 ; répétitions locales (« les filles, les filles »). |

- Réglage raisonnable pour ce modèle : température autour de 0,8, avec Top-K ou Top-P.
- Le modèle choisit bien le registre selon le début : un dialogue (tiret) donne des répliques, une description donne du récit. Mais le fil se perd après 2 ou 3 phrases : le même constat que pendant l'entraînement (contexte de 128, 5,5 M de paramètres).
- Le premier essai de l'étape 3 est identique à celui du Top-P de l'étape 2 : même graine et mêmes réglages, c'est normal.
- Les boucles à basse température sont la signature d'un petit modèle : il n'a aucun souvenir de ce qu'il vient de dire au-delà de la fenêtre et retombe sur la réponse la plus fréquente.
- Règle d'or des expériences : **une seule variable à la fois** (corpus, architecture ou hyperparamètres).

## 11. Parler au modèle (leçon 10)

```
.venv/bin/python src/06_corpus/scripts/10_chat.py
```

- Boucle interactive sur le CPU : modes `chat` (réplique de dialogue) et `suite`, commandes `/temp`, `/topk`, `/topp`, `/longueur`, `/nouveau`, `/quitter`.
- Sert à comprendre la différence entre **entraînement** (les poids changent) et **utilisation** (poids figés, rien n'est retenu de la conversation, fenêtre de 128 tokens). Détail dans `src/06_corpus/fiches/10_chat.md`.
- Appareil : `APPAREIL = "cpu"` par défaut ; `"cuda"` pour comparer les temps. Le calcul est trop petit pour que le GPU aide.
- Ce n'est pas un assistant : il continue des romans, il ne répond pas aux questions.
- Test de vérification (entrées simulées) : fonctionne, environ 0,2 à 0,4 s par réponse sur le CPU.
- Premiers essais de Jérôme : « bonjour » → « Être à la tête » ; « que sait tu faire ? » → une réplique sans rapport avec la question (« "Hannah" est une question d'importance… ») ; « ok » → « "Tu es un mot "l'"sil" ? ». Verdict : « n'importe quoi », ce qui est normal : le modèle n'a jamais vu de paires question → réponse, seulement des romans.

## Journal : le modèle apprend à parler (étape 9, premier entraînement)

Toutes les 500 étapes, le script demande au modèle de continuer « Il était une fois » (50 tokens, température 0,8). Résultats réels du premier entraînement, comme un bébé qui apprend à parler :

| Étape | Validation (perplexité) | Ce que le modèle écrit | Ce qu'il a appris |
|---|---|---|---|
| 0 (essai de 40 étapes) | 7,51 (1824) | « Il était une foisig Cor évogar quar mul atte cafév… » | rien : des morceaux au hasard |
| 500 | 4,79 (121) | « Les avêcher atteinvière. Tu n'aurait-il ? » | des syllabes, des mots inventés |
| 1000 | 4,28 (72) | « - Je crois qui y aura de toi. - C'est que je devais bien. » | les tirets de dialogue, des phrases courtes |
| 2000 | 3,96 (52) | « - Oui, il n'y a pas connu. - Parce que je vous parlais, fit E. » | la structure d'un roman (`- réplique, fit X`) |
| 3000 | 3,81 (45) | « Il s'était tourné lentement, mais il avait eu les petits certitudes… » | des phrases longues, le passé du récit |
| 4500 | 3,69 (40) | « Un matin, une fois à l'heure, il n'y avait rien de plus violent, il n'avait rien eu pour dîner… » | un style de roman, mais pas de fil d'une phrase à l'autre |

Observations :

- Après l'étape 1000 la progression ralentit (de 0,03 à 0,05 de perte gagnée par tranche de 500 étapes) : rendements décroissants, car les gains faciles (orthographe, ponctuation, dialogues) sont pris les premiers, et le taux d'apprentissage descend en cosinus (de 1e-3 à 1e-4 : grands pas au début, petits pas pour affiner à la fin). Les gains restants, comme le sens et le fil de l'histoire, sont plus difficiles.
- Pas de cohérence d'une phrase à l'autre : contexte de 128 tokens (environ 100 mots), 5,5 millions de paramètres (GPT-2 en a 124 millions), 1,2 passage sur le corpus, et l'entraînement ne demande que « prédis le token suivant ».
- Pistes pour progresser, une variable à la fois : modèle plus gros (le GPU n'utilise que 1,4 Go sur 8,6 Go), contexte plus long, plus d'étapes, meilleur nettoyage, meilleur tokenizer.
- En moins de 4 minutes de GPU, le modèle passe du charabia à de vraies phrases françaises : orthographe, conjugaison, ponctuation, dialogues. Il n'a reçu aucune règle de grammaire : seulement la consigne « prédis le token suivant ».
- Il sait écrire, pas raconter : chaque phrase est plausible, mais elles ne se suivent pas (contexte de 128 tokens, 5,5 millions de paramètres).
- Il invente encore des noms et des mots (« Jüvenu », « Honald », « croûtre »).
- Train et validation restent proches (3,55 et 3,69) : pas de surapprentissage à ce stade.

Texte à comparer plus tard avec les modèles plus gros : garder le même début « Il était une fois » et les mêmes réglages de génération.

### Résultat final du premier entraînement (20 000 étapes, 14 minutes)

Sortie complète du script, sans modification : `logs/entrainement_livres_sortie.txt`.

| Étape | Train | Validation | Perplexité |
|---|---|---|---|
| 500 | 5,831 | 4,794 | 121 |
| 5 000 | 3,520 | 3,663 | 39 |
| 10 000 | 3,326 | 3,484 | 33 |
| 15 000 | 3,199 | 3,374 | 29 |
| 20 000 | 3,141 | **3,321** | **28** |

- Une nouvelle meilleure validation à chaque mesure : le modèle progressait encore à la fin, mais à tout petits pas (0,001 à 0,005 par tranche de 500 étapes sur les dernières).
- Train (3,14) et validation (3,32) restent proches : pas de surapprentissage. Le modèle est limité par sa taille et sa durée, pas par le corpus.
- De 5 000 à 20 000 étapes : perplexité de 39 à 28 seulement, pour trois fois plus de temps.
- Textes de la fin : dialogues et récit plausibles phrase par phrase (« Il a baissé les yeux vers eux. - Ne m'avez pas fallu. »), mais sans fil ; encore quelques mots inventés (« humestrée », « Helitticus », « la mercie »). Un échantillon (étape 8000) recopie un fragment de table des matières (« 2273-Oastronomie........ ») : les mentions de mise en page restent dans le corpus.
- Les caractères étranges venaient surtout de ces mots inventés et de caractères rares du vocabulaire (611 caractères) ; à tester seul : réduire les caractères rares ou nettoyer plus.

## Où j'en suis (10 novembre 2026, fin de journée) et pour reprendre

Fait et lancé : étapes 1 à 8 (346 livres, BPE de 4 612 tokens, `train.bin` de 66 M de tokens, validation sur 17 livres).

Étape 9 terminée (`07_entrainement_corpus.py`, 20 000 étapes, 14 minutes sur la GTX 1070) : validation finale 3,321 (perplexité 28) contre 8,44 au hasard. Sortie complète dans `logs/entrainement_livres_sortie.txt`. Le GPU n'utilisait que 1,4 Go sur 8,6 Go : de la marge pour un modèle plus gros.

Étape 10 : leçons 8 (génération), 9 (courbe) et 10 (chat) écrites et lancées. Sorties : `logs/generation_livres_sortie.txt`, `logs/courbe_entrainement_livres.png`. Fiches `08`, `09`, `10` écrites, README à jour.

Dernières décisions du 10 novembre :
- Le chat (`10_chat.py`) tourne sur le CPU par défaut : réglage `APPAREIL = "cpu"` ou `"cuda"` en tête du script. Les deux fonctionnent (test : environ 0,1 s sur CPU, 0,3 s sur GPU pour la première réponse, à cause du démarrage du GPU) ; pour un si petit modèle le GPU n'accélère pas. `scores_suivant` de `05_inference/tools/generation.py` utilise maintenant l'appareil du modèle.
- Le chat répond « n'importe quoi » : normal, le modèle ne sait que continuer des romans. Un vrai assistant demanderait un second entraînement sur des paires question → réponse.

À faire demain :
1. Lancer le chat avec `APPAREIL = "cuda"` et comparer les temps affichés (optionnel).
2. Première expérience à une seule variable : modèle plus gros (d, blocs) ou contexte T plus long, avec les mêmes réglages de génération pour comparer.
3. Autres pistes, **une seule variable à la fois** : plus d'étapes, vocabulaire du BPE (611 caractères dont beaucoup de rares), nettoyage plus poussé (pages de titre, mentions d'éditeur, tables des matières).
4. Mettre à jour `.github/copilot-instructions.md` avec le chapitre `06_corpus` (à proposer).
5. Rien n'est commité : relire avant de commiter (les livres, textes et `.bin` sont ignorés par git ; `logs/*.csv`, `logs/*.txt` et `logs/*.png` ne le sont pas, à décider).

Suivi des expériences (fiche technique, analyse, tableau) : [`suivi_experiences_corpus.md`](suivi_experiences_corpus.md).

Pour retrouver cette conversation : `copilot --resume`, ou la commande `/resume` dans le CLI.

## Pièges rencontrés

- Les `.azw3` ne sont pas des doublons des `.mobi`.
- Le drapeau des octets parasites en fin de bloc se lit à l'octet `0xF2` du record 0 (pas à `16 + 0xF2`).
- La langue déclarée dans le fichier est souvent fausse : se fier à la langue estimée.
- Comparer la diversité du vocabulaire seulement entre livres de longueur proche.
- Le triage a besoin de la colonne `fichier` : relancer `03_metadonnees.py` si `metadata.csv` date d'avant.
- Le BPE lent recompte toutes les paires à chaque fusion : inutilisable sur le corpus, d'où `bpe_rapide.py`.
- Ex æquo entre deux paires : le BPE lent et le rapide ne choisissent pas dans le même ordre, donc les fusions diffèrent un peu sans que la qualité change.
