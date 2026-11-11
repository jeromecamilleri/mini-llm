# Installation de PyTorch

Journal de l'installation de PyTorch dans le venv du projet (nécessaire à la leçon 9 et à la partie PyTorch).

## État de départ

| vérification | résultat |
|---|---|
| `import torch` dans `.venv`, `python3`, `/usr/bin/python3` | `ModuleNotFoundError` |
| `pip list` / environnements conda | aucun PyTorch |
| contenu de `.venv` | NumPy, Matplotlib et leurs dépendances seulement |
| GPU | GTX 1070 (architecture Pascal, capacité 6.1) |
| pilote NVIDIA | 580.178.04, CUDA 13.0 |
| Python du venv | 3.14.4 |
| espace disque libre | 257 Go |

## Choix

- Version **CUDA 12.6** (`cu126`) plutôt que la version par défaut : les versions CUDA récentes retirent le support des GPU Pascal comme la GTX 1070. Le pilote 580 reste compatible avec cu126.
- Installation dans le venv du projet, sans toucher au Python système.

## Commande lancée

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
```

(lancée le 2026-11-10 à 16:21 ; téléchargement de plusieurs Go)

## Résultat

Installation réussie : `torch-2.14.1+cu126`, avec ses dépendances (bibliothèques `nvidia-*-cu12`, `triton`, `sympy`, `networkx`, `jinja2`, `fsspec`, `filelock`, etc.).

## Vérification

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

| test | résultat |
|---|---|
| version | `2.14.1+cu126` |
| `torch.cuda.is_available()` | `True` |
| GPU détecté | NVIDIA GeForce GTX 1070, capacité (6, 1) |
| architectures compilées | `sm_50` à `sm_90` (le code `sm_60` fonctionne sur la capacité 6.1) |
| produit de matrices 1000×1000 sur la GPU | fonctionne |

## `requirements.txt`

Ajouts :

- `--extra-index-url https://download.pytorch.org/whl/cu126` (source des roues CUDA 12.6) ;
- `torch==2.14.1+cu126`.

Les dépendances de PyTorch (`nvidia-*`, `triton`, `sympy`…) ne sont pas épinglées une par une : pip les résout à partir de `torch`.

## Pour réinstaller

```bash
.venv/bin/pip install -r requirements.txt
```
