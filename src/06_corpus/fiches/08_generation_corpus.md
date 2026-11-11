# 08 — Faire écrire le modèle

Script : [`../scripts/08_generation_corpus.py`](../scripts/08_generation_corpus.py)

## L'essentiel en 30 secondes

- On recharge `livres_meilleur.pt` : les poids ne changent pas, on règle seulement **comment choisir** le token suivant.
- Trois réglages (fonctions de `05_inference`) : la **température**, le **Top-K** et le **Top-P**.
- Même graine pour tous les essais : si le texte change, c'est à cause du réglage et de rien d'autre.

## Ce que le modèle a écrit (« Il était une fois », 120 tokens)

| Réglage | Résultat |
|---|---|
| température 0,3 | texte correct mais **boucle** : « - Je ne sais pas. » sept fois |
| température 0,8 | phrases variées et plausibles, sans fil entre elles |
| température 1,2 | **déraille** : mots tronqués ou inventés (« survavant », « défav ») |
| 0,8 + Top-K 40 | le récit le plus continu (à la première personne) |
| 0,8 + Top-P 0,9 | proche de 0,8, avec quelques répétitions (« les filles, les filles ») |

## À observer quand tu lances le script

- À basse température, le modèle choisit toujours le favori : il retombe sur la réplique la plus fréquente et boucle.
- À haute température, il tire des tokens très improbables et le texte se défait.
- Le début change le registre : un tiret donne des répliques, une description donne du récit.
- Le premier essai de l'étape 3 est identique à celui du Top-P de l'étape 2 : même graine, mêmes réglages.

## Ce qu'on en déduit

- Un bon compromis pour ce modèle : température autour de 0,8 avec Top-K ou Top-P.
- Les boucles à basse température sont la signature d'un petit modèle : au-delà de sa fenêtre de 128 tokens, il ne se souvient de rien.
- Les réglages de génération améliorent le style, pas la cohérence : pour ça, il faut un modèle plus gros ou un contexte plus long (expériences à tester une variable à la fois).
