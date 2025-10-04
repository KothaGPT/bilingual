# MyPy Fixes Guide

This guide lists actionable fixes for current MyPy warnings in the repository, grouped by category, with minimal, non-invasive changes you can apply incrementally.

Use it as a checklist. All examples reference real files in this repo.

---

## 0) Recommended MyPy configuration

Update `pyproject.toml` to a supported Python version and set sane defaults:

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
no_implicit_optional = true
check_untyped_defs = true
warn_redundant_casts = true
warn_unused_ignores = true
# Optionally relax strictness for placeholder modules
# ignore_missing_imports = true
```

If you prefer targeted ignores, add `mypy.ini` with per-module overrides:

```ini
[mypy]
python_version = 3.9

[mypy-sacrebleu.*]
ignore_missing_imports = True

[mypy-rouge_score.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-transformers.*]
ignore_missing_imports = True
```

---

## 1) tokenizer: None has no attribute "load"

File: `bilingual/tokenizer.py`

Root cause: variable (e.g., `spm` or `tokenizer`) can be `None` before `.load()`.

Fix options:
- Add a guard before calling `.load()`
- Or assert non-None

Example:
```python
from typing import Optional

spm: Optional[SentencePieceProcessor] = None
...
if spm is None:
    spm = SentencePieceProcessor()
spm.load(model_path)
```

If a function returns tokenizer instance:
```python
def load_tokenizer(path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    ok = sp.load(path)
    if not ok:
        raise ValueError(f"Failed to load tokenizer: {path}")
    return sp
```

---

## 2) Return type unions too wide (lists of lists vs flat lists)

Files: `bilingual/tokenizer.py` (encode), `bilingual/api.py` (tokenize)

Root cause: functions sometimes return `List[List[int]]` or `List[List[str]]` when type hints say `List[int] | List[str]`.

Fix options:
- Normalize return type based on a flag (`return_ids`, `batch`) and reflect it in overloads.

Example with overloads:
```python
from typing import List, overload

@overload
def tokenize(text: str, *, return_ids: False = False) -> List[str]: ...
@overload
def tokenize(text: str, *, return_ids: True) -> List[int]: ...

def tokenize(text: str, *, return_ids: bool = False):
    tokens = _tokenize_impl(text)
    return tokens if not return_ids else _ids(tokens)
```

For batched inputs, add separate API (e.g., `tokenize_batch`) and type it to return nested lists.

---

## 3) `normalize_unicode(form=...)` expects Literal

File: `bilingual/normalize.py`

Root cause: `unicodedata.normalize` expects a literal value among `'NFC'|'NFD'|'NFKC'|'NFKD'`.

Fix:
```python
from typing import Literal

Form = Literal['NFC','NFD','NFKC','NFKD']

def normalize_unicode(text: str, form: Form = 'NFC') -> str:
    return unicodedata.normalize(form, text)
```

Ensure callers pass only those literals.

---

## 4) Returning Any from functions declared to return str

Files: `bilingual/models/translate.py`, `bilingual/models/lm.py`, `bilingual/evaluation.py`

Root cause: placeholder stubs returning results from untyped libs.

Quick fix for placeholders:
```python
from typing import cast

result = some_untyped_generate(...)
return cast(str, result)
```

Prefer real typing where possible; otherwise, narrow with `cast` or make return type `str | None` if appropriate (and handle None).

---

## 5) `api.py`: return-value union mismatch

File: `bilingual/api.py` (`tokenize` et al.)

Align signature with actual behavior (single vs batched):
- Single text → `List[str]` or `List[int]`
- Batch texts → `List[List[str]]` or `List[List[int]]`

Introduce a batched API:
```python
def tokenize_batch(texts: List[str], *, return_ids: bool = False) -> List[List[int]] | List[List[str]]:
    ...
```

Keep `tokenize(text: str, ...) -> List[str] | List[int]` for single inputs.

---

## 6) `api.py`: untyped var needs annotation

Error: `Need type annotation for "flags"`

Fix:
```python
from typing import List

flags: List[str] = []
```

---

## 7) Path vs str mismatches in datasets

File: `bilingual/data_utils.py`

Root cause: parameters typed as `str` are used as `Path`.

Fix:
```python
from typing import Union
from pathlib import Path

PathLike = Union[str, Path]

def __init__(..., file_path: Optional[PathLike] = None):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    self.file_path: Optional[Path] = file_path
```

Also update helper methods to accept/return `Path` consistently, and adjust calls.

---

## 8) Missing import stubs for third-party libs

File: `bilingual/evaluation.py` (sacrebleu, rouge_score, sklearn)

Options:
- Add per-module ignores (preferred for optional deps):
  ```python
  from sacrebleu import corpus_bleu  # type: ignore[import-not-found]
  ```
- Or set per-module `ignore_missing_imports = True` (see config above).

---

## 9) Type errors in CLI

File: `bilingual/cli.py`

Root cause: CLI code assigns dicts/strings into variables typed as `List[str] | List[int]` (or untyped), and indexes a list using `str` keys.

Fix strategy:
- Don’t reuse the same `result` variable across different command branches.
- Keep variables local per branch with precise types.
- Avoid indexing a list with a string key (e.g., `result["text"]`).

Example pattern:
```python
if args.command == "tokenize":
    tokens: List[str] | List[int] = bb.tokenize(
        args.text, tokenizer=args.tokenizer, return_ids=args.ids
    )
    print(tokens)
elif args.command == "readability":
    report: dict[str, Any] = bb.readability_check(args.text, lang=args.lang)
    print(report["score"])  # dict access ok
```

---

## 10) Returning Any in evaluation metrics

File: `bilingual/evaluation.py`

Wrap untyped returns with `float(...)` or `cast(float, ...)`:
```python
from typing import cast

score = compute_some_metric(...)
return cast(float, float(score))
```

For placeholder values, return a concrete `float`:
```python
return 0.0
```

Also fix: `results["num_samples"]` is `int`; do not reassign to `str`.

---

## 11) Incremental workflow

Apply fixes in this order to reduce churn:
- **A.** Config: bump `python_version` to 3.9
- **B.** Path/str mismatches in `data_utils.py`
- **C.** Normalize literal types in `normalize.py`
- **D.** API/Tokenizer return type consistency (single vs batch)
- **E.** CLI branch-local variables and types
- **F.** Add `ignore_missing_imports` for optional metric libs
- **G.** Add `cast(...)` for placeholder returns

Run:
```bash
mypy bilingual/ --show-error-codes
```

---

## 12) Quick patches (copy/paste)

- Literal form type:
```python
from typing import Literal
Form = Literal['NFC','NFD','NFKC','NFKD']
```

- PathLike helper:
```python
from typing import Union
from pathlib import Path
PathLike = Union[str, Path]
```

- Typed list var:
```python
from typing import List
flags: List[str] = []
```

- Casting placeholder return:
```python
from typing import cast
return cast(str, value)
```

- Ignore missing import (surgical):
```python
from sacrebleu import corpus_bleu  # type: ignore[import-not-found]
```

---

## 13) Defer strict typing for placeholder modules

It’s acceptable to keep placeholders with `cast(...)` or limited ignores until real models are integrated. Keep CI `continue-on-error: true` for MyPy.

---

## 14) Tracking

Create an issue "Tighten type checking" and check off items A–G above as you address them. This keeps type improvements incremental and non-blocking.
