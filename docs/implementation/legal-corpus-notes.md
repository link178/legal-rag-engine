# Legal corpus adapter — Phase 13 (`legalize-*`)

## Goal

Provide an **operator-controlled**, **local-first** adapter that imports a directory shaped like `legalize-es/` / `legalize-eu/` style corpora (or any folder tree of supported files). It:

- discovers supported files recursively;
- attaches **stable JSON metadata** (`corpus_adapter`, `corpus_name`, `corpus_relative_path`, jurisdiction heuristics, etc.);
- optionally parses **minimal Markdown frontmatter** (scalar `key: value` lines only — **no PyYAML** dependency);
- reuses **`IngestionService`** + **`ingest_file_persisted`** for persistence and idempotency.

This phase **does not** add HTTP APIs, Streamlit flows, remote GitHub sync, OCR, or retrieval-side metadata filters.

## Supported formats

Same as core ingestion:

- `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf`

## CLI

```bash
python -m app.ingestion.adapters.legal_corpus.cli data/sample_corpus/legalize_sample --json
python -m app.ingestion.adapters.legal_corpus.cli data/sample_corpus/legalize_sample --persist --json
python -m app.ingestion.adapters.legal_corpus.cli data/sample_corpus/legalize_sample \
  --json --markdown-report reports/legalize_import.md
```

Flags:

| Flag | Meaning |
|------|---------|
| `--persist` | Persist each file via `ingest_file_persisted` (Postgres + migrations required). |
| `--json` | Print a stable summary JSON (`sort_keys=True`). |
| `--markdown-report PATH` | Write a short Markdown report. |
| `--limit N` | Process only the first `N` files after deterministic ordering. |
| `--fail-fast` | Stop on first failed file; umbrella run marked **failed** when persisting. |

### Exit codes

- `1` if `--persist` and **zero** files discovered.
- `1` if `--fail-fast` and any item **`failed`**.
- `1` on CLI usage errors (missing path, not a directory) or unexpected exceptions during import.

## Discovery rules

Implemented in `app/ingestion/adapters/legal_corpus/discovery.py`:

- Recursive walk from the corpus root.
- Only listed extensions (**case-insensitive** suffix).
- Skips **symlinks**.
- Skips **hidden files** (`name` starts with `.`).
- Skips subtrees under ignored directory names: `.git`, `.venv`, `node_modules`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `dist`, `build`.
- Skips intermediate directories whose name starts with `.`.
- Deterministic order: POSIX `relative_path` ascending.

## Metadata contract

Each imported document receives extra metadata merged via `IngestionService.ingest_file(..., extra_metadata=...)` **before** `file_checksum_sha256` is applied (checksum remains authoritative).

Typical keys:

| Key | Notes |
|-----|-------|
| `corpus_adapter` | Constant `"legalize"`. |
| `corpus_name` | Root directory name (`legalize_sample`, etc.). |
| `corpus_relative_path` | POSIX path relative to corpus root. |
| `source_family` | `"legalize"` (alias for adapter family). |
| `jurisdiction` | Heuristic `es` / `eu` / `unknown`; frontmatter overrides. |
| `language` | Frontmatter `language` if present; else `es` when jurisdiction is `es`; else `unknown`. |
| `legal_document_type` | Filename heuristic (`regulation`, `directive`, `law`, `royal_decree`, `guidance`, …); frontmatter `document_type` overrides. |
| `canonical_id`, `effective_date`, `version` | Optional; usually from frontmatter. |

**Important:** frontmatter **`title`** is copied into metadata for convenience; **`Document.title`** still comes from loaders (e.g. first Markdown `#` heading).

## Frontmatter parser

- Starts at file beginning with `---` line; closes at next `---` line.
- Only single-line `key: value` scalars.
- Supports quoted strings, booleans (`true`/`false`), and integer values (digits only).
- Invalid lines warn per document but **do not** abort the corpus run.

### Known limitation

The YAML block remains inside `Document.raw_text` for this phase — values are mirrored into metadata but **not stripped** from body text.

## Persistence & traceability

When `--persist`:

1. One umbrella **`processing_runs`** row is created with `run_type="corpus_import"` (`running` → `completed` or `failed`).
2. Each file triggers the existing **`ingest`** run (`run_type="ingest"`) via `ingest_file_persisted`.
3. Child runs include `metadata_json["corpus_run_id"]` referencing the umbrella UUID string.

Idempotency matches **`(source_path, checksum)`** as today: re-importing unchanged files yields **`reused`** items without rewriting metadata.

## Sample corpus

Synthetic fixtures live under:

`data/sample_corpus/legalize_sample/`

## Tests

- Default **`pytest`** stays DB-free for unit coverage (`tests/unit/test_legal_corpus_*.py`).
- Opt-in integration: `LEGAL_RAG_RUN_INTEGRATION_DB=1` → `tests/integration/test_legal_corpus_persist.py`.

## Explicit non-goals (Phase 13)

- Remote repo clone / GitHub downloads / crawling / scraping.
- OCR or scanned-PDF recovery beyond existing loader errors.
- New FastAPI routes or Streamlit corpus UI.
- Automatic legislation updates or compliance workflows.
- Semantic dedupe beyond `(source_path, checksum)`.
