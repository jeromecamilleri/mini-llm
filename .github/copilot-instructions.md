# Copilot instructions — mini-llm

Pedagogical lab to build a small GPT-style model from scratch and measure how the training corpus affects it. The full plan (labs 1–7, experiments A–D, metrics) is in `docs/Readme.md` (French). Code, comments, docstrings and docs are written in **French**; keep that language.

## Running code

No build, lint or test suite exists. Scripts are run directly with the project venv (Linux, GTX 1070):

```bash
.venv/bin/python src/01_fundamentals/scripts/a_la_main/01_neurone.py   # lesson (NumPy, by hand)
.venv/bin/python src/01_fundamentals/scripts/pytorch/09_pytorch.py     # lesson (PyTorch)
.venv/bin/python src/01_fundamentals/tools/perceptron.py
.venv/bin/python src/02_tokenizer/scripts/01_caracteres.py
.venv/bin/python src/03_transformer/scripts/07_petit_gpt.py            # GPT complet (NumPy)
.venv/bin/python src/04_training/scripts/03_checkpoints.py             # entraîne et écrit models/checkpoints/*.pt
.venv/bin/python src/05_inference/scripts/02_echantillonnage.py        # lit ces checkpoints
```

Order matters: `05_inference` lessons need the checkpoints written by `04_training/scripts/03_checkpoints.py`. The first launch of `.venv/bin/python` sometimes fails with `No module named torch`; simply rerun it.

- Dependencies: `pip install -r requirements.txt` (pinned; NumPy, Matplotlib, PyTorch cu126 for the GTX 1070; install log in `docs/installation_pytorch.md`).
- Folders under `src/` start with a number (not importable as Python packages): lessons add their sibling `tools/` folder to `sys.path` (`Path(__file__).resolve().parents[N] / "tools"`) and import modules directly (`from forward import DenseNetwork`).
- Matplotlib scripts use `matplotlib.use("Agg")` (headless) and save figures to `logs/` instead of showing them.

## Architecture

Layout under `src/`, numbered in learning order: `01_fundamentals/` (labs 1–2), `02_tokenizer/` (lab 3), `03_transformer/` (lab 4, NumPy: attention to a full small GPT), `04_training/` (lab 6, PyTorch GPT, training loop, checkpoints), `05_inference/` (lab 7, generation, temperature/Top-K/Top-P, model comparison), `06_corpus/` (346 Kindle books: extraction, cleaning, BPE, tokenization, training, chat, embeddings; experiments tracked in `docs/suivi_experiences_corpus.md`). Each folder has a `README.md` with its lesson table. Always refer to chapters by folder name (`02_tokenizer`), never « Labo N »: the numbers do not match the folders. Data flows `datasets/originals → extracted → cleaned → tokenized`; models go to `models/checkpoints` then `models/final`; each experiment writes its curve and output to `logs/` and its checkpoints to `models/checkpoints/` (tracked in `docs/suivi_experiences_corpus.md`).

Each `src/NN_*/` folder has three kinds of files:
- `tools/`: library-style modules (e.g. `activations.py`, `forward.py`, `backpropagation.py`, `optimiseurs.py`, `vocabulary.py`) with a docstring stating the lab and the formula.
- `scripts/` (in `01_fundamentals`, split into `a_la_main/` for NumPy and `pytorch/`): `NN_*.py` didactic lessons run top-to-bottom, long explanatory docstring, stepwise `ÉTAPE n` sections, no `assert`.
- Cross-chapter imports: a lesson may add another chapter's `tools/` to `sys.path` (e.g. `05_inference/tools/generation.py` imports `04_training/tools/`). The shared mini-corpus is `src/02_tokenizer/corpus_animaux.txt` (15 sentences, 67 words) and is reused by chapters 02 to 05.
- `fiches/` (same sub-folders): one `NN_*.md` sheet per lesson, same name as the lesson, linking to it, with worked example, tables, figures and a final « Ce qu'on en déduit » section. No `.py` exercises.

## Conventions

- Lessons are for a learner: define every term at its first occurrence, open each sheet with « L'essentiel en 30 secondes », keep the same running example (« le chat dort . ») and simple French. When the learner says he did not understand something, explain it in chat then add the explanation to both the sheet and the lesson docstring.
- Do not commit on your own; leave changes uncommitted for review.
- Books, datasets (including `datasets/cleaned_v2/`), `.bin` files and `models/checkpoints/*` are git-ignored (copyrighted/heavy): add any new data folder to `.gitignore` BEFORE creating it, and check `git status` before every commit. `logs/*.csv|txt|png` are small and tracked. Illustrations live in `img/`.
- Build core mechanisms by hand (NumPy first, then PyTorch tensors); no prefab GPT architectures, pretrained models, high-level training frameworks, or tokenizer libraries that hide the mechanics.
- Never modify source datasets in `datasets/originals`; extraction/cleaning must write new files.
- For experiments, change only one variable at a time (corpus, architecture, or hyperparameters).
- Keep GTX 1070 memory limits in mind (small models, modest batch sizes).
