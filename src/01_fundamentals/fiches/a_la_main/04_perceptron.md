# Exercices : un seul neurone face au XOR

Cours associé : [`../../scripts/a_la_main/04_perceptron.py`](../../scripts/a_la_main/04_perceptron.py) (module : [`../../tools/perceptron.py`](../../tools/perceptron.py))

Même exemple que [`03_propagation_avant.md`](03_propagation_avant.md) : le XOR sur `(x1, x2)`, cibles `[0, 1, 1, 0]`.
Mais ici on utilise **un seul neurone** (celui de [`01_neurone.md`](01_neurone.md)), pas un réseau `[2, 4, 1]`.

Question : un neurone seul peut-il apprendre le XOR ?

## Exemple résolu

Sorties du neurone pour `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]` (cibles `0 1 1 0`) et MSE (voir la fiche 03 pour sa définition) :

| w | b | sorties | MSE |
|---|---|---------|-----|
| [0, 0] | 0 | 0.5, 0.5, 0.5, 0.5 | 0.2500 |
| [5, 5] | −2.5 | 0.076, 0.924, 0.924, 0.999 | 0.2540 |
| [10, 10] | −5 | 0.007, 0.993, 0.993, 1.000 | 0.2500 |
| [5, −5] | 0 | 0.500, 0.007, 0.993, 0.500 | 0.3717 |

Puis une recherche à l'aveugle sur toutes les combinaisons entières `(w1, w2, b)` de −8 à 8 :
meilleur réglage `(0, 0, 0)`, **MSE = 0.25**.

## Ce qu'on en déduit

- Les réglages `[5, 5]` et `[10, 10]` font un **OU** : bons sur 3 cas, mais `[1, 1]` sort ≈ 1 au lieu de 0.
- Chaque réglage corrige des cas et en rate d'autres.
- Le meilleur réglage possible est « ne rien savoir » (sortie 0.5 partout), soit la MSE du **hasard (0.25)**.
- **Un neurone seul ne peut pas apprendre le XOR** : il sépare les cas par une droite, et le XOR n'est pas séparable par une droite.
- C'est pourquoi la leçon 3 utilise une **couche cachée** : plusieurs neurones combinés peuvent tracer plusieurs droites.
- Cela rejoint un fait historique : en 1969, Minsky et Papert ont montré que le perceptron (un neurone) ne sait pas faire le XOR, ce qui a ralenti la recherche. Les réseaux multicouches entraînés par rétropropagation ont levé cette limite.

## Prochaine étape

On a un réseau (leçon 3) et une perte (MSE). Il reste à **entraîner** : la rétropropagation calcule comment chaque poids influence la perte, puis la descente de gradient les corrige. Aucun des deux n'est encore écrit.
