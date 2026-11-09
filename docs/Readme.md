# Mini-LLM Laboratory

## Programme pédagogique et expérimental

**Version :** 1.0\
**Date :** 8 novembre 2026\
**Environnement cible :** CamUbuntu, Ubuntu Linux, GPU NVIDIA GeForce
GTX 1070\
**Répertoire de travail :** `/dataSSD/travailJerome/mini-llm`

------------------------------------------------------------------------

## 1. Présentation du projet

L'objectif est de construire progressivement un petit modèle de langage
de type GPT, entraîné localement, afin de comprendre concrètement les
mécanismes qui permettent à un LLM d'apprendre à partir d'un corpus et
de générer du texte.

Le projet privilégie la compréhension et l'expérimentation plutôt que
l'utilisation de composants « tout faits ».

Il ne s'agit pas de reproduire un modèle industriel comme GPT, Llama ou
Mistral. Le but est de construire un modèle miniature, suffisamment
simple pour pouvoir en examiner les composants, les calculs et les
résultats.

Une attention particulière sera portée à l'influence des données
d'entraînement : domaine, volume, qualité, diversité et répétitions du
corpus.

### Principes directeurs

-   Comprendre chaque mécanisme avant de l'utiliser.
-   Programmer nous-mêmes les composants pédagogiques importants.
-   Éviter les bibliothèques qui masquent toute l'architecture.
-   Utiliser PyTorch principalement pour les tenseurs, les opérations
    GPU et l'autodifférentiation lorsque cela devient pertinent.
-   Conserver des expériences reproductibles et comparer les résultats.
-   Mesurer les résultats avec des indicateurs quantitatifs et des
    exemples générés.

------------------------------------------------------------------------

## 2. Matériel et stockage disponibles

### Configuration connue

  Élément                 Configuration
  ----------------------- -------------------------
  Machine                 CamUbuntu
  Système                 Ubuntu Linux
  GPU                     NVIDIA GeForce GTX 1070
  Mémoire vidéo           8 Go
  Stockage de travail     `/dev/sdc1`, ext4
  Point de montage        `/dataSSD`
  Capacité du volume      Environ 440 Go
  Espace libre constaté   Environ 258 Go

Autres volumes repérés :

-   `/` : partition ext4 d'environ 93 Go.
-   `/home` : partition ext4 d'environ 98 Go.
-   `/dev/sdd` : disque dur de 2 To, formaté en NTFS.
-   Deux autres SSD Samsung de 480 Go, avec des partitions NTFS.

### Dimensionnement

Les 258 Go libres sur `/dataSSD` sont largement suffisants pour les
premiers travaux.

Une enveloppe indicative peut être réservée ainsi :

  Usage                                  Budget indicatif
  ------------------------------------ ------------------
  Données sources et corpus préparés                20 Go
  Modèles et checkpoints                            20 Go
  Environnements et outils                          20 Go
  Marge pour les expérimentations                   40 Go
  **Total planifié**                           **100 Go**

Ces valeurs sont des budgets de confort, pas des consommations prévues
exactes. Il restera de la place pour les évolutions du projet.

La GTX 1070 permet d'envisager l'entraînement de petits modèles. Sa
génération ne dispose pas de Tensor Cores ; il faudra donc adapter la
taille du modèle, la longueur des séquences et les lots d'entraînement.
Les limites réelles seront mesurées pendant les premiers essais.

------------------------------------------------------------------------

## 3. Organisation du dépôt

Le laboratoire sera regroupé sous `/dataSSD/travailJerome/mini-llm`.

``` text
/dataSSD/travailJerome/mini-llm/
├── datasets/
│   ├── originals/       # Fichiers sources conservés intacts
│   ├── extracted/       # Textes extraits des sources
│   ├── cleaned/         # Textes nettoyés
│   └── tokenized/       # Corpus convertis en identifiants
│
├── experiments/
│   ├── french/          # Expériences sur corpus français
│   ├── technical/       # Expériences sur corpus technique
│   ├── mixed/           # Corpus mixtes
│   └── comparisons/     # Comparaisons contrôlées
│
├── models/
│   ├── checkpoints/     # États intermédiaires d'entraînement
│   └── final/           # Modèles retenus
│
├── src/
│   ├── fundamentals/    # Neurones et réseaux simples
│   ├── tokenizer/       # Tokenisation
│   ├── transformer/    # Attention et architecture GPT
│   ├── training/        # Boucles d'entraînement
│   └── inference/       # Génération de texte
│
├── notebooks/           # Explorations et visualisations
├── logs/                # Journaux d'exécution
└── docs/                # Documentation du projet
```

Création de l'arborescence :

``` bash
mkdir -p /dataSSD/travailJerome/mini-llm/{datasets/{originals,extracted,cleaned,tokenized},experiments/{french,technical,mixed,comparisons},models/{checkpoints,final},src/{fundamentals,tokenizer,transformer,training,inference},notebooks,logs,docs}
```

Puis :

``` bash
cd /dataSSD/travailJerome/mini-llm
```

Les fichiers sources ne doivent jamais être modifiés directement. Les
étapes d'extraction et de nettoyage produisent de nouveaux fichiers.

------------------------------------------------------------------------

## 4. Parcours pédagogique

Le parcours est organisé en sept laboratoires successifs. Chaque étape
doit produire du code exécutable, des observations et une courte
documentation.

### Laboratoire 1 --- Fondamentaux du Deep Learning

**Objectif :** comprendre comment un neurone artificiel et un réseau
apprennent à ajuster leurs paramètres.

Notions :

-   Entrées, poids et biais.
-   Somme pondérée.
-   Fonctions d'activation : ReLU, Sigmoid, Tanh et Leaky ReLU.
-   Propagation avant (*forward propagation*).
-   Fonction de perte (*loss function*).
-   Dérivées, gradients et règle de dérivation en chaîne.
-   Rétropropagation (*backpropagation*).
-   Descente de gradient et optimiseur Adam.

Travail pratique :

-   Programmer un neurone avec NumPy.
-   Construire un petit réseau dense.
-   Calculer manuellement la propagation avant.
-   Implémenter la rétropropagation sur un petit exemple.
-   Comparer les gradients calculés manuellement à ceux obtenus par
    PyTorch.
-   Observer l'évolution de la fonction de perte.

Fichiers envisagés :

``` text
src/fundamentals/
├── perceptron.py
├── activations.py
├── forward.py
└── backpropagation.py
```

**Résultat attendu :** comprendre comment une erreur de prédiction est
transformée en corrections des poids du réseau.

### Laboratoire 2 --- Réseau neuronal avec PyTorch

**Objectif :** retrouver les mêmes principes avec des tenseurs et
commencer à utiliser le GPU.

Notions :

-   Tenseurs et dimensions.
-   Opérations matricielles.
-   Dataset et DataLoader.
-   Fonction de perte.
-   Calcul automatique des gradients.
-   Optimiseur et boucle d'entraînement.
-   Séparation entraînement / validation.

Travail pratique :

-   Reprendre un réseau simple avec PyTorch.
-   Comparer CPU et GPU.
-   Observer les dimensions des tenseurs à chaque étape.
-   Sauvegarder et recharger un modèle.

**Résultat attendu :** maîtriser une boucle d'entraînement sans utiliser
de framework de haut niveau qui cache l'architecture.

### Laboratoire 3 --- Comprendre la représentation du langage

**Objectif :** comprendre comment du texte devient une séquence de
nombres exploitable par un réseau neuronal.

Notions :

-   Caractères, mots, sous-mots et tokens.
-   Vocabulaire.
-   Tokenisation.
-   Encodage et décodage des identifiants.
-   BPE (*Byte Pair Encoding*).
-   Embeddings.
-   Fenêtre de contexte et séquences d'entraînement.

Travail pratique :

1.  Commencer par un tokenizer caractère par caractère.
2.  Construire un vocabulaire simple.
3.  Implémenter un tokenizer élémentaire.
4.  Étudier le fonctionnement de BPE.
5.  Comparer la taille du vocabulaire et le nombre de tokens produits.
6.  Visualiser les vecteurs d'embedding appris.

Un tokenizer BPE pourra être implémenté à des fins pédagogiques. Une
bibliothèque spécialisée pourra ensuite servir de référence pour
comparer les résultats.

Fichiers envisagés :

``` text
src/tokenizer/
├── vocabulary.py
├── tokenizer_basic.py
└── bpe.py
```

**Résultat attendu :** comprendre la transformation du corpus en
séquences d'identifiants et son influence sur la taille des données et
le modèle.

### Laboratoire 4 --- Construire un Transformer

**Objectif :** programmer progressivement l'architecture qui constitue
le cœur d'un GPT.

Notions :

-   Embeddings de tokens.
-   Encodage positionnel ou embeddings de position.
-   Query, Key et Value.
-   Produit scalaire et scores d'attention.
-   Softmax.
-   Self-Attention.
-   Masque causal.
-   Multi-Head Attention.
-   Réseau Feed Forward.
-   Connexions résiduelles (*residual connections*).
-   Normalisation (*LayerNorm*).
-   Empilement de blocs Transformer.

Travail pratique :

-   Programmer une attention simple sur une petite séquence.
-   Afficher les scores et poids d'attention.
-   Ajouter un masque causal et vérifier qu'un token ne voit pas le
    futur.
-   Construire une tête d'attention, puis plusieurs têtes.
-   Assembler un bloc Transformer.
-   Empiler plusieurs blocs pour former un petit GPT.

Fichiers envisagés :

``` text
src/transformer/
├── embeddings.py
├── attention.py
├── transformer_block.py
└── gpt.py
```

**Résultat attendu :** disposer d'une architecture GPT miniature dont
les principales opérations sont explicites et compréhensibles.

### Laboratoire 5 --- Préparer les corpus

**Objectif :** constituer des jeux de données propres, documentés et
comparables.

Sources envisagées :

-   Environ 200 livres présents dans la bibliothèque Kindle.
-   Textes libres de droits.
-   Documentation technique et informatique dont les conditions de
    réutilisation le permettent.
-   Autres corpus publics avec licence compatible.

Pour chaque source, conserver des métadonnées :

-   Identifiant du document.
-   Titre et auteur, si disponibles.
-   Langue.
-   Catégorie ou domaine.
-   Source et licence.
-   Taille du fichier.
-   Nombre de caractères et de mots.
-   Nombre de tokens après tokenisation.

Étapes :

1.  Inventorier les fichiers.
2.  Vérifier les droits et licences.
3.  Extraire le texte lorsque cela est autorisé.
4.  Nettoyer les artefacts de mise en page.
5.  Détecter les doublons et répétitions.
6.  Séparer les corpus par domaine.
7.  Constituer les ensembles d'entraînement, de validation et de test.
8.  Tokeniser les textes.
9.  Documenter précisément la composition de chaque corpus.

**Point juridique :** les fichiers Kindle peuvent être protégés par DRM.
Ne pas contourner une protection technique. Utiliser uniquement des
fichiers et des usages dont l'extraction et l'entraînement sont
autorisés. Vérifier également les licences des corpus publics.

**Résultat attendu :** plusieurs corpus identifiés et reproductibles,
prêts pour les expériences.

### Laboratoire 6 --- Entraîner notre mini-GPT

**Objectif :** entraîner un modèle de langage autorégressif qui prédit
le token suivant.

Notions :

-   Découpage des textes en séquences.
-   Entrées et cibles décalées d'un token.
-   Prédiction du prochain token.
-   Logits et distribution de probabilités.
-   Cross Entropy.
-   Rétropropagation.
-   AdamW.
-   Batch size.
-   Learning rate.
-   Nombre d'époques.
-   Checkpoints.
-   Validation et perplexité.

La boucle d'entraînement devra être écrite explicitement :

1.  Charger un lot de séquences.
2.  Fournir les tokens d'entrée au modèle.
3.  Calculer les prédictions.
4.  Comparer les prédictions aux tokens attendus.
5.  Calculer la Cross Entropy.
6.  Calculer les gradients.
7.  Mettre à jour les paramètres.
8.  Enregistrer périodiquement les métriques et checkpoints.
9.  Évaluer le modèle sur des données de validation.

**Résultat attendu :** obtenir un modèle entraîné localement et capable
de générer du texte token après token.

### Laboratoire 7 --- Inférence et expérimentation

**Objectif :** observer le comportement du modèle et comprendre les
facteurs qui l'influencent.

Notions :

-   Génération autorégressive.
-   Température.
-   Top-K et Top-P.
-   Longueur du contexte.
-   Stratégies d'échantillonnage.
-   Évaluation qualitative et quantitative.

Travail pratique :

-   Générer du texte à partir de prompts identiques.
-   Comparer plusieurs modèles entraînés.
-   Modifier la température et les paramètres de génération.
-   Comparer les résultats avec les métriques d'entraînement et de
    validation.

**Résultat attendu :** comprendre les capacités et limites du modèle, et
relier les résultats aux données et paramètres utilisés.

------------------------------------------------------------------------

## 5. Le cœur expérimental : mesurer l'influence du corpus

Cette partie est centrale dans le projet.

L'idée est de construire plusieurs corpus et d'entraîner des modèles
comparables pour observer ce que les données font réellement apprendre
au réseau.

### Expérience A --- Influence du domaine

Constituer par exemple :

  Corpus             Contenu
  ------------------ -----------------------------------------
  Français général   Romans, nouvelles et essais
  Informatique       Linux, réseaux, programmation, systèmes
  Scientifique       Mathématiques et sciences
  Mixte              Mélange des domaines précédents

Questions :

-   Le vocabulaire généré change-t-il ?
-   Le style de génération évolue-t-il ?
-   Le modèle reproduit-il mieux les structures propres à un domaine ?
-   Les connaissances apparentes sont-elles liées à la fréquence des
    sujets dans le corpus ?

### Expérience B --- Influence du volume

Entraîner des modèles sur des volumes croissants, par exemple :

-   1 million de tokens.
-   5 millions de tokens.
-   10 millions de tokens.
-   50 millions de tokens, si les ressources le permettent.

Il faudra distinguer le volume de données disponible du nombre de tokens
effectivement présentés au modèle pendant l'entraînement.

Questions :

-   La loss de validation diminue-t-elle ?
-   À partir de quel volume les gains deviennent-ils faibles ?
-   Le modèle mémorise-t-il certains passages ?
-   La diversité du corpus compte-t-elle davantage que son volume brut ?

### Expérience C --- Influence de la qualité

Comparer des corpus :

-   Nettoyés et structurés.
-   Contenant des erreurs typographiques.
-   Avec beaucoup de répétitions.
-   Hétérogènes ou bruités.

Questions :

-   Les répétitions favorisent-elles la mémorisation ?
-   Les erreurs sont-elles reproduites dans les générations ?
-   Le nettoyage améliore-t-il la généralisation ?

### Expérience D --- Influence de la diversité linguistique

Comparer :

-   Français uniquement.
-   Français et anglais.
-   Plusieurs langues.

Questions :

-   Le modèle conserve-t-il les capacités dans chaque langue ?
-   Le mélange des langues modifie-t-il la génération ?
-   Le vocabulaire partagé est-il efficace ?

### Règle méthodologique essentielle

Pour attribuer une différence au corpus, il faut maintenir les autres
paramètres constants autant que possible :

-   Même architecture.
-   Même nombre de paramètres.
-   Même tokenizer pour les expériences qui isolent l'effet du corpus.
-   Même nombre de tokens d'entraînement lorsque l'on compare la qualité
    ou le domaine.
-   Même longueur de contexte.
-   Mêmes hyperparamètres.
-   Même protocole de validation et de test.
-   Plusieurs graines aléatoires lorsque le coût le permet.

Pour comparer le volume, le nombre de tokens d'entraînement est
volontairement la variable étudiée. Pour comparer le contenu ou la
qualité, il faut au contraire contrôler ce nombre.

Les ensembles de validation et de test doivent être séparés avant la
constitution des séquences d'entraînement afin de limiter les fuites de
données. Les documents quasi identiques doivent également être repérés.

------------------------------------------------------------------------

## 6. Mesures et indicateurs

Chaque expérience doit produire des résultats quantitatifs et
qualitatifs.

  -----------------------------------------------------------------------
  Indicateur                          Interprétation
  ----------------------------------- -----------------------------------
  Training loss                       Erreur sur les données
                                      d'entraînement

  Validation loss                     Erreur sur des données non
                                      utilisées pour ajuster les poids

  Perplexity                          Mesure dérivée de la loss,
                                      exprimant la difficulté de
                                      prédiction

  Nombre de tokens                    Volume de texte effectivement
                                      traité

  Nombre de paramètres                Taille du réseau

  Temps d'entraînement                Coût de calcul

  Mémoire GPU utilisée                Contrainte matérielle

  Exemples générés                    Analyse qualitative du comportement
  -----------------------------------------------------------------------

La perplexité est généralement calculée comme l'exponentielle de la
Cross Entropy moyenne, lorsque celle-ci est exprimée en logarithmes
naturels. Les comparaisons doivent utiliser le même tokenizer et le même
protocole d'évaluation.

Il faudra tracer les courbes de training loss et validation loss en
fonction des étapes ou des époques.

Un écart croissant entre les deux peut signaler du surapprentissage.

Les exemples générés devront être conservés avec leur prompt, le modèle
utilisé et les paramètres de génération.

------------------------------------------------------------------------

## 7. Choix techniques

  Technologie   Rôle
  ------------- --------------------------------------------------
  Python        Langage principal
  NumPy         Premiers calculs et implémentations pédagogiques
  PyTorch       Tenseurs, GPU, gradients et entraînement
  Matplotlib    Graphiques et courbes d'apprentissage
  Jupyter       Exploration et visualisation
  Git           Historique du code et des expériences

### Ce que nous voulons éviter au départ

-   Utiliser directement une architecture GPT préfabriquée.
-   Confier toute la tokenisation à une bibliothèque sans en comprendre
    le fonctionnement.
-   Utiliser un modèle pré-entraîné en prétendant l'avoir entraîné
    nous-mêmes.
-   Masquer la boucle d'entraînement derrière un framework de haut
    niveau.
-   Modifier simultanément le corpus, l'architecture et les
    hyperparamètres.

Des bibliothèques pourront être utilisées ensuite comme références ou
pour des tâches périphériques, mais les mécanismes centraux resteront
visibles.

### Contraintes GPU

La GTX 1070 dispose de 8 Go de VRAM. Elle est adaptée à de petites
expériences, mais les performances dépendront fortement :

-   Du nombre de paramètres.
-   De la longueur de contexte.
-   Du batch size.
-   De la précision numérique.
-   De l'implémentation des opérations.

On commencera avec un modèle très réduit, puis on augmentera
progressivement sa taille. Il ne faut pas fixer une taille maximale
théorique avant d'avoir mesuré la mémoire et les performances sur la
machine.

------------------------------------------------------------------------

## 8. Environnement logiciel : vérifications préalables

Avant d'installer quoi que ce soit, relever l'état actuel de la machine
:

``` bash
python3 --version
nvidia-smi
nvcc --version
```

La commande `nvcc` peut être absente si le CUDA Toolkit n'est pas
installé. Ce n'est pas nécessairement bloquant : les distributions
PyTorch adaptées peuvent embarquer les bibliothèques CUDA nécessaires à
leur fonctionnement.

Il faudra ensuite choisir une version de PyTorch compatible avec le
pilote NVIDIA et la GTX 1070. Vérifier la compatibilité de
l'architecture GPU avant de figer l'environnement.

Créer un environnement Python isolé dans le projet, par exemple avec
`venv`, afin de ne pas modifier les paquets système :

``` bash
cd /dataSSD/travailJerome/mini-llm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Les dépendances seront ajoutées progressivement et enregistrées dans un
fichier de dépendances versionné.

------------------------------------------------------------------------

## 9. Première feuille de route opérationnelle

  ------------------------------------------------------------------------
                         Ordre Action                État
  ---------------------------- --------------------- ---------------------
                             1 Créer l'arborescence  À faire
                               du projet             

                             2 Vérifier le GPU, le   À faire
                               pilote et Python      

                             3 Créer l'environnement À faire
                               Python isolé          

                             4 Programmer un neurone À faire
                               avec NumPy            

                             5 Programmer un petit   À faire
                               réseau et la          
                               rétropropagation      

                             6 Comparer avec PyTorch À faire

                             7 Construire un         À faire
                               tokenizer simple      

                             8 Programmer            À faire
                               l'attention et le     
                               Transformer           

                             9 Inventorier les       À faire
                               livres et vérifier    
                               les droits            
                               d'utilisation         

                            10 Préparer plusieurs    À faire
                               corpus                

                            11 Entraîner le premier  À faire
                               mini-GPT              

                            12 Comparer les effets   À faire
                               des corpus            
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 10. Critères de réussite

Le projet sera réussi si, à son terme, nous sommes capables de :

1.  Expliquer le calcul effectué par un neurone et par une couche.
2.  Comprendre et vérifier une rétropropagation.
3.  Expliquer le rôle des embeddings et de la tokenisation.
4.  Programmer une attention causale et un bloc Transformer.
5.  Entraîner un petit modèle de langage sans modèle pré-entraîné.
6.  Générer du texte localement.
7.  Reproduire une expérience à partir de sa configuration et de ses
    données.
8.  Comparer objectivement plusieurs corpus.
9.  Expliquer les limites du modèle et les effets observés.

Le résultat recherché n'est pas un assistant aussi performant qu'un LLM
moderne. C'est une compréhension pratique de la chaîne complète :
données, représentation, architecture, apprentissage et génération.

------------------------------------------------------------------------

## 11. Prochaine étape

La première séance consistera à :

1.  Vérifier la configuration logicielle et GPU de CamUbuntu.
2.  Créer le dépôt Git et l'environnement Python.
3.  Programmer un premier neurone artificiel avec NumPy.
4.  Faire varier les poids et le biais pour observer leur effet.
5.  Construire une première fonction de perte et visualiser son
    évolution.

**Règle du laboratoire : chaque notion doit être comprise, expérimentée
et mesurée avant de passer à la suivante.**
