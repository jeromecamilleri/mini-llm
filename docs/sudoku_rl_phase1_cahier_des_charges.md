# Cahier des charges technique — Sudoku RL, Phase 1
## Environnement, Oracle, Récompenses, Dataset

> **Périmètre** : ce document couvre exclusivement la Phase 1. Pas de réseau neuronal ici.  
> Le réseau (Phase 2) sera construit dans `src/07_sudoku_rl/` en reprenant les patterns de `src/01_fundamentals/` et `src/04_training/`.

---

## 0. Positionnement dans le cours

Le projet Sudoku RL est un **chapitre 07** qui s'appuie sur tout ce qui existe :

| Ce qu'on réutilise | D'où ça vient | Comment |
|---|---|---|
| Pattern `nn.Module` + boucle d'entraînement | `src/04_training/tools/gpt_torch.py`, `02_entrainement.py` | La boucle RL remplace `tirer_batch` mais garde `zero_grad / backward / step` |
| `checkpoint.py` | `src/04_training/tools/checkpoint.py` | Repris tel quel pour sauvegarder le réseau de politique |
| Adam à la main | `src/01_fundamentals/tools/optimiseurs.py` | Référence pédagogique ; on utilisera `torch.optim.Adam` en Phase 2 |
| Concept de validation / arrêt anticipé | `src/04_training/scripts/02_entrainement.py` | On mesure le taux de résolution au lieu de la perte cross-entropy |

Le GPT n'est **pas** réutilisé architecturalement. On garde la *mécanique* (boucle, checkpoints, outils NumPy) mais pas l'architecture séquentielle language-model.

---

## 1. Structure des dossiers

```
src/
  07_sudoku_rl/
    README.md
    fiches/                    ← une fiche .md par leçon (même convention)
    scripts/
      01_environnement.py      ← Phase 1, leçon 1 : l'env et l'oracle
      02_dataset.py            ← Phase 1, leçon 2 : chargement et curriculum
      (03_..., 04_... Phase 2)
    tools/
      sudoku_env.py            ← SudokuEnv  (contrat de l'environnement)
      oracle.py                ← Oracle     (contrat de l'oracle)
      sudoku_utils.py          ← helpers partagés (validation, affichage)
      dataset.py               ← SudokuDataset (chargement, itération)

datasets/
  sudoku/
    raw/                       ← fichiers source bruts (non versionnés)
    sudoku_dataset.csv         ← dataset normalisé (voir §4)
    sudoku_dataset_mini.csv    ← 1 000 grilles pour dev rapide
```

---

## 2. L'environnement Sudoku — `SudokuEnv`

### 2.1 Représentation de l'état

Une grille 9 × 9 est représentée par **deux tenseurs** :

```
grille      : np.ndarray shape (9, 9), dtype int8
              valeurs : 0 = case vide, 1–9 = chiffre placé
grille_init : np.ndarray shape (9, 9), dtype int8   (copie immuable)
              les cases ≠ 0 sont les cases fixes (données de départ)
```

Le réseau recevra en entrée (Phase 2) un tenseur dérivé de `grille` ; ici on fixe uniquement ce que l'environnement expose.

**Entrée pour le réseau (specification, pas implémentée en Phase 1) :**
```
shape (81, 10)  — 81 cases × 10 canaux one-hot
  canal 0      : 1 si la case est vide, 0 sinon
  canaux 1–9   : 1 si la case contient le chiffre k, 0 sinon
```
Ce format permet une convolution ou un MLP sans pré-traitement supplémentaire.

### 2.2 Espace des actions

729 actions possibles = 81 cases × 9 chiffres.

```python
action = ligne * 9 * 9 + colonne * 9 + (chiffre - 1)
# avec ligne, colonne ∈ [0, 8] et chiffre ∈ [1, 9]
```

Fonction inverse fournie par `sudoku_utils.decode_action(action) -> (ligne, col, chiffre)`.

### 2.3 Contrat public de `SudokuEnv`

```python
class SudokuEnv:
    # --- Construction ---
    def __init__(self, grille_init: np.ndarray, solution: np.ndarray):
        """
        grille_init : grille de départ (cases vides = 0, cases fixes ≠ 0)
        solution    : solution complète connue (utilisée par l'oracle uniquement)
        Les cases fixes ne peuvent jamais être écrasées.
        """

    # --- Interface principale ---
    def reset(self) -> np.ndarray:
        """Remet la grille à son état initial. Retourne l'état."""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Applique l'action et retourne (état, récompense, terminé, info).

        état    : np.ndarray (9,9) après l'action
        récompense : float (voir §3)
        terminé : True si grille complète et correcte, ou si n_coups_max atteint
        info    : dict avec clés 'oracle_verdict', 'n_coups', 'n_cases_restantes'
        """

    # --- Propriétés utilitaires ---
    @property
    def n_cases_vides(self) -> int: ...

    @property
    def n_coups(self) -> int: ...

    def actions_valides(self) -> list[int]:
        """
        Liste des actions qui ne placent pas dans une case déjà remplie.
        (Ne filtre PAS selon les règles du Sudoku : c'est volontaire.)
        """

    def render(self) -> str:
        """Représentation ASCII de la grille (pour le débogage)."""
```

### 2.4 Règles de terminaison

| Condition | `terminé` | Explication |
|---|---|---|
| Toutes les cases vides sont remplies **correctement** | `True` | Victoire |
| `n_coups >= n_coups_max` | `True` | Timeout (défaut : `n_coups_max = 500`) |
| Toutes les autres situations | `False` | L'épisode continue |

La grille est déclarée **correcte** ssi chaque ligne, chaque colonne et chaque bloc 3 × 3 contient exactement les chiffres 1 à 9 (sans doublons). C'est la seule vérification finale ; on ne compare pas à `solution` pour la terminaison.

---

## 3. L'oracle — `Oracle`

### 3.1 Contrat

```python
class Oracle:
    def __init__(self, solution: np.ndarray):
        """
        solution : solution complète de la grille en cours.
        L'oracle est lié à une seule grille ; on en crée un par épisode.
        """

    def verifier(self, grille: np.ndarray, ligne: int, col: int, chiffre: int) -> str:
        """
        Retourne 'OUI', 'NON' ou 'PEUT_ETRE'.

        Algorithme :
          1. Si la case (ligne, col) est déjà remplie dans la grille → 'NON'
             (action impossible : on ne réécrit pas une case)
          2. Si le chiffre crée un doublon dans la ligne, la colonne
             ou le bloc 3×3 → 'NON'
          3. Si le chiffre correspond à solution[ligne, col] → 'OUI'
          4. Sinon → 'PEUT_ETRE'
             (légalement posable, mais pas la bonne réponse)
        """
```

### 3.2 Signification précise des trois réponses

| Verdict | Ce que ça signifie | Exemple |
|---|---|---|
| `NON` | Contradiction immédiate avec les règles **ou** action impossible | Doublon en ligne, case déjà remplie |
| `OUI` | Le chiffre est celui de la solution connue | `solution[r, c] == chiffre` **et** pas de doublon |
| `PEUT_ETRE` | Légalement posable, mais pas la bonne réponse selon la solution | Un 5 autorisé dans la case, mais la solution y met un 3 |

**Contrainte de pureté** : `Oracle` ne connaît que `solution`. Il n'a ni stratégie, ni propagation de contraintes, ni solver interne.  
Le réseau ne voit jamais `solution` directement.

### 3.3 Pourquoi `PEUT_ETRE` existe-t-il ?

Sans solution de référence, on ne peut pas distinguer « légalement posable mais faux » de « juste ». Le verdict `PEUT_ETRE` évite de récompenser positivement un coup légal mais incorrect, ce qui créerait un signal trompeur pendant l'apprentissage.

---

## 4. Les récompenses

### 4.1 Tableau de récompenses

| Événement | Récompense | Justification |
|---|---|---|
| Oracle répond `OUI` | **+1.0** | Signal positif clair |
| Oracle répond `PEUT_ETRE` | **-0.1** | Pénalité légère : le coup est légal mais coûte du temps |
| Oracle répond `NON` (doublon) | **-1.0** | Violation des règles |
| Action impossible (case déjà remplie) | **-1.0** | Même signal que doublon |
| Grille entièrement résolue (terminaison victoire) | **+10.0** | Bonus de fin d'épisode |
| Timeout (`n_coups >= n_coups_max`) | **0.0** | Pas de pénalité de mort ; on arrête proprement |

### 4.2 Règles de calcul

- La récompense est retournée par `step()` **avant** de tester la terminaison.  
  Autrement dit : si le dernier coup correct déclenche la victoire, le réseau reçoit `+1.0` (coup correct) **puis**, dans le même `step()`, `terminé=True` et le bonus `+10.0` est **additionné** à la récompense retournée : récompense finale = `+1.0 + 10.0 = +11.0`.
- Pas de reward shaping supplémentaire en Phase 1. On pourra l'envisager en Phase 3 si l'apprentissage stagne.

### 4.3 Paramètre `n_coups_max`

Valeur par défaut : `500` (bien au-delà des 81 coups nécessaires, pour ne pas punir l'exploration). Configurable à la construction de `SudokuEnv`.

---

## 5. Le dataset — `SudokuDataset`

### 5.1 Format du fichier CSV

```
puzzle,solution,difficulte
003020600900305001001806400008102900700000008006708200002609500800203009005010300,483921657967345821251876493548132976729564138136798245372689514814253769695417382,facile
...
```

| Colonne | Type | Description |
|---|---|---|
| `puzzle` | str (81 chars) | Chiffres 1–9 et `0` pour les cases vides, ligne par ligne, gauche→droite |
| `solution` | str (81 chars) | Même format, sans zéro |
| `difficulte` | str | `'facile'`, `'moyen'`, `'difficile'` (voir §5.3) |

### 5.2 Contrat de `SudokuDataset`

```python
class SudokuDataset:
    def __init__(self, chemin_csv: str):
        """Charge le CSV et indexe les grilles par difficulté."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict:
        """
        Retourne {
            'puzzle'    : np.ndarray (9, 9), dtype int8,
            'solution'  : np.ndarray (9, 9), dtype int8,
            'difficulte': str,
            'id'        : int  (index dans le CSV, pour reproductibilité)
        }
        """

    def par_difficulte(self, niveau: str) -> 'SudokuDataset':
        """Vue filtrée : dataset.par_difficulte('facile')."""

    def melanger(self, graine: int = 42) -> 'SudokuDataset':
        """Réordonne les grilles (pour varier l'ordre d'entraînement)."""
```

### 5.3 Critères de difficulté

La difficulté est estimée par le **nombre de cases vides** (simple, déterministe, reproductible) :

| Niveau | Cases vides | Typique |
|---|---|---|
| `facile` | 25 – 35 | Sudoku "débutant" |
| `moyen` | 36 – 46 | Sudoku "intermédiaire" |
| `difficile` | 47 – 58 | Sudoku "expert" |

Cette classification est appliquée **à la génération du dataset normalisé** via un script `scripts/02_dataset.py`. Elle peut être affinée en Phase 3 si le curriculum learning le nécessite.

### 5.4 Source recommandée

[Sudoku Exchange Puzzle Bank](https://sudokuexchange.com/panel/) propose des fichiers CSV avec puzzle + solution.  
Le script `scripts/02_dataset.py` télécharge, valide et normalise vers le format §5.1.

**Validation minimale à l'import** (dans `SudokuDataset.__init__`) :
- Exactement 81 caractères dans `puzzle` et `solution`.
- `solution` sans zéro.
- `solution` cohérente avec `puzzle` (les cases données dans `puzzle` doivent correspondre à `solution`).
- `solution` valide (pas de doublon en ligne/colonne/bloc).  
  Toute ligne invalide est ignorée avec un warning.

---

## 6. Le module `sudoku_utils`

Fonctions partagées entre `SudokuEnv`, `Oracle` et `SudokuDataset` :

```python
# Conversion action ↔ coordonnées
def encode_action(ligne: int, col: int, chiffre: int) -> int: ...
def decode_action(action: int) -> tuple[int, int, int]: ...  # (ligne, col, chiffre)

# Validation
def est_valide(grille: np.ndarray) -> bool:
    """True si aucun doublon dans les 9 lignes, 9 colonnes et 9 blocs."""

def est_complete(grille: np.ndarray) -> bool:
    """True si aucune case vide ET est_valide()."""

def a_doublon(grille: np.ndarray, ligne: int, col: int, chiffre: int) -> bool:
    """True si placer chiffre en (ligne, col) crée un doublon."""

def bloc_indices(ligne: int, col: int) -> tuple[int, int]:
    """Retourne (ligne_debut, col_debut) du bloc 3×3 contenant (ligne, col)."""

# Parsing CSV ↔ ndarray
def chaine_vers_grille(s: str) -> np.ndarray: ...  # "003020..." → (9,9)
def grille_vers_chaine(g: np.ndarray) -> str: ...  # (9,9) → "003020..."

# Affichage
def afficher_grille(grille: np.ndarray) -> str:
    """Représentation ASCII avec séparateurs de blocs."""
```

---

## 7. Script de leçon 1 — `scripts/01_environnement.py`

Suit la convention du cours : script linéaire, lisible de haut en bas, avec des étapes commentées.

```
ÉTAPE 1 : charger une grille du dataset mini
ÉTAPE 2 : créer l'environnement et l'oracle
ÉTAPE 3 : jouer quelques coups à la main (OUI, NON, PEUT_ETRE)
ÉTAPE 4 : vérifier la terminaison sur une grille pré-résolue
ÉTAPE 5 : afficher les statistiques (n_coups, récompenses cumulées)
```

Le script doit tourner **sans GPU**, sans télécharger de données, avec `sudoku_dataset_mini.csv` (1 000 grilles incluses dans le repo pour les tests).

---

## 8. Script de leçon 2 — `scripts/02_dataset.py`

```
ÉTAPE 1 : télécharger le dataset source (ou lire depuis raw/)
ÉTAPE 2 : valider chaque ligne (§5.4)
ÉTAPE 3 : classifier par difficulté (§5.3)
ÉTAPE 4 : écrire sudoku_dataset.csv
ÉTAPE 5 : afficher la distribution (nb facile / moyen / difficile)
```

---

## 9. Ce que la Phase 1 ne contient PAS

| Élément | Reporté à |
|---|---|
| Le réseau neuronal (CNN + Policy + Value) | Phase 2 |
| La boucle d'entraînement RL (PPO ou REINFORCE) | Phase 2 |
| Le curriculum learning (sélection dynamique de difficulté) | Phase 3 |
| L'export ONNX / TFLite | Phase 4 |
| L'application Android | Phase 4 |
| Reward shaping avancé | Phase 3 |

---

## 10. Critères de validation de la Phase 1

La Phase 1 est terminée quand tous ces tests passent :

```
[ ] est_valide() retourne True sur une grille correcte connue
[ ] est_valide() retourne False si on introduit un doublon
[ ] Oracle.verifier() retourne 'NON' pour un doublon
[ ] Oracle.verifier() retourne 'OUI' pour le bon chiffre (sans doublon)
[ ] Oracle.verifier() retourne 'PEUT_ETRE' pour un chiffre légal mais faux
[ ] SudokuEnv.step() retourne terminé=True après avoir rempli toute la grille correctement
[ ] SudokuEnv.step() retourne terminé=True après n_coups_max dépassé
[ ] SudokuDataset charge le CSV mini et retourne des dicts bien formés
[ ] scripts/01_environnement.py tourne sans erreur et affiche les 5 étapes
[ ] scripts/02_dataset.py tourne et produit sudoku_dataset.csv avec 3 niveaux
```

---

## 11. Dépendances

Aucune dépendance nouvelle requise pour la Phase 1 ; le projet utilise déjà `numpy` et `torch`.  
Le script `02_dataset.py` peut utiliser `requests` (disponible dans la plupart des environnements Python) pour télécharger les données source. Pas d'ajout à `requirements.txt` nécessaire.

---

## Annexe : pourquoi ne pas utiliser l'architecture GPT

Le GPT résout le problème suivant : *étant donné une séquence de tokens, quel est le token suivant ?*  
Il optimise une perte de **prédiction supervisée** (cross-entropy) sur un corpus fixe.

Le Sudoku RL résout : *étant donné un état de jeu, quelle action maximise la récompense cumulée ?*  
Il optimise une **espérance de récompense** dans un environnement dynamique.

Les différences structurelles sont fondamentales :

| | GPT (cours existant) | Sudoku RL |
|---|---|---|
| Signal d'apprentissage | Supervision dense (un label par token) | Récompense clairsemée (surtout en fin d'épisode) |
| Données | Corpus fixe, indépendant de l'entraînement | Générées dynamiquement par l'interaction |
| Architecture | Transformer (attention causale) | CNN + têtes Policy/Value (sans attention) |
| Objectif | Minimiser cross-entropy | Maximiser récompense cumulée |
| Évaluation | Perte de validation, perplexité | Taux de grilles résolues |

Les deux projets partagent la **mécanique de base** (gradient, optimiseur, checkpoints) mais pas l'architecture ni l'objectif. C'est ce qui rend la comparaison pédagogiquement intéressante.
