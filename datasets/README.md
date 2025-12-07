# Datasets

This directory contains all datasets used in the bilingual project, including raw sources, intermediate processing outputs, and final training/evaluation splits.

## Layout

- `wikipedia/`
  - Wikipedia dumps and processed data for Bangla / bilingual language modelling.
  - See `datasets/wikipedia/README.md` for full details and workflow.
- `processed/`
  - Generic workspace for dataset processing pipelines (raw → cleaned → filtered → final).
  - Can be reused by multiple scripts and experiments.

## Conventions

- **Text format**: UTF-8, one example per line unless otherwise documented.
- **Splits**: Prefer `train/`, `val/`, `test/` subdirectories.
- **File naming**: Use `<lang>_<split>.txt` for simple text datasets (e.g. `bn_train.txt`).
- **Paths**: Scripts generally assume paths relative to the project root (e.g. `datasets/wikipedia/raw`).

## Wikipedia Dataset

The Wikipedia LM dataset lives in `datasets/wikipedia/` and is managed by the helper scripts in `scripts/` and `Makefile.wiki`.

Key entry points:

- `make -f Makefile.wiki download-bn` – download Bangla dumps to `datasets/wikipedia/raw/`.
- `make -f Makefile.wiki preprocess` – extract, clean, and split into `datasets/wikipedia/processed/`.
- `make -f Makefile.wiki validate` – run quality checks on `datasets/wikipedia/processed/`.
- `make -f Makefile.wiki prepare-hf` – build a Hugging Face dataset under `datasets/wikipedia/hf_dataset/`.

See `datasets/wikipedia/README.md` for more details.

## Adding a New Dataset

When adding a new dataset (e.g. for a downstream task):

1. **Create a directory** under `datasets/`, e.g. `datasets/my_task/`.
2. **Document it** in a `README.md` inside that directory:
   - Source, license, languages
   - Directory structure (raw/processed/splits)
   - Expected formats
3. **Follow naming conventions** where possible:
   - `raw/`, `processed/` (with `train/`, `val/`, `test/`)
   - Text files in UTF-8, one example per line.
4. **Optional metadata**: Add a `dataset_meta.json` or `.yaml` describing schema, splits, and sizes.

This keeps datasets discoverable and reusable across scripts and experiments.
