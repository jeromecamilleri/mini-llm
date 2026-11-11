# 10 — Parler au modèle : entraînement contre utilisation

Script : [`../scripts/10_chat.py`](../scripts/10_chat.py)

## L'essentiel en 30 secondes

- Entraînement et utilisation sont deux étapes différentes avec le même modèle : on a **appris** les poids (leçon 7), on s'en **sert** maintenant.
- Pendant l'utilisation, les poids sont figés : rien n'est corrigé, rien n'est retenu de la conversation.
- Ce n'est pas un vrai assistant : il a appris à continuer des romans, pas à répondre à des questions.

## Entraînement et utilisation

| | Entraînement (leçon 7) | Utilisation (leçon 10) |
|---|---|---|
| but | apprendre les poids | se servir des poids appris |
| poids | changent à chaque étape | figés (`modele.eval()`) |
| texte donné | des fenêtres tirées des livres | ton texte |
| réponse attendue | connue : le token suivant du livre | inconnue : le modèle la devine |
| perte, AdamW | oui | non |
| coût | 14 minutes de GPU | une fraction de seconde de CPU |
| résultat | un fichier `.pt` | du texte |

## Les deux modes

- **chat** : ta phrase devient une réplique (`- ta phrase`) et le modèle écrit la réplique suivante, parce que les romans contiennent beaucoup de dialogues.
- **suite** : le modèle continue ton texte.
- Commandes : `/chat`, `/suite`, `/temp 0.8`, `/topk 40`, `/topp 0.9`, `/longueur 80`, `/nouveau`, `/memoire 0` (sans mémoire) ou `/memoire 1`, `/quitter` (valeur 0 pour désactiver Top-K ou Top-P).

## À observer quand tu lances le script

- Le compteur « tokens en mémoire » : au-delà de 128, les plus anciennes répliques sont oubliées.
- Pose la même phrase plusieurs fois : la réponse change, car le tirage est aléatoire (comme à l'étape 2 de la leçon 8, mais sans graine fixe).
- Essaie une question (« Quelle est la capitale de la France ? ») : il écrit une réplique plausible, rarement une vraie réponse. Il imite des dialogues, il ne sait rien « répondre ».
- `/nouveau` efface la mémoire : le modèle n'a pas changé, seule la fenêtre est vidée.

## Le modèle recopie ses propres réponses

Mesures sur le modèle A (température 0,6, Top-K 40) :

| Cas | Probabilité |
|---|---|
| « je te dis » après « pourquoi », conversation neuve | 0,0001 (1 sur 10 000) |
| « je te dis » après « il fait beau », conversation neuve | 0,00005 (1 sur 20 000) |
| même réponse, avec la 1re réponse « je te dis » dans la mémoire | 0,055 (5,5 %) |
| « Ô » en premier mot, sans mémoire | 1,5 % |
| « Ô » en premier mot, après trois réponses en « Ô » | 98 % |

Deux réponses identiques dans deux conversations indépendantes : environ 1 chance sur 200 millions. Dans la même conversation, c'est 5,5 % : le modèle voit sa réponse précédente et la recopie (apprentissage en contexte, version rudimentaire). Baisser la température renforce la boucle. Les « Ô » viennent de *Zarathoustra*, du théâtre et de textes anciens du corpus. Remèdes : `/nouveau`, `/memoire 0`, température plus haute.

## Ce qu'on en déduit

- Ce qui est dans la mémoire du chat oriente fortement la suite, sans toucher aux poids.
- Ce que le modèle sait est entièrement dans les poids ; ta conversation ne les modifie jamais.
- Pour un vrai assistant, il faudrait un second entraînement, sur des exemples question → réponse (ajustement par instructions). Pas dans ce projet pour l'instant.
