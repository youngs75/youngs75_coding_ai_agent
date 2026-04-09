<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# loop1_dataset

## Purpose
Handles the entire Loop 1 process from synthetic data generation through human review integration to Golden Dataset finalization.

## Key Files
| File | Description |
|------|-------------|
| `synthesizer.py` | Generates synthetic QA from corpus documents |
| `csv_exporter.py` | Converts synthetic JSON → review CSV |
| `csv_importer.py` | Converts review CSV → golden JSON |
| `feedback_augmenter.py` | Augments expected_output based on reviewer feedback |
| `golden_builder.py` | Loop1 orchestration |
| `prompts.py` | Prompt constants used for Loop1 augmentation |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Ensure changes do not break data format field compatibility (`id`, `input`, `expected_output`, `context`, `source_file`).
- When modifying CSV columns, update `scripts/02_export_for_review.py`, `scripts/03_import_reviewed.py`, and documentation together.
- Validate both the `skip_review` path and `reviewed_csv_path` path simultaneously to prevent regressions.

### Testing Requirements
- `.venv/bin/pytest tests/test_synthesizer.py tests/test_loop1_synthesizer_generation.py -q`
- When CSV schema changes, verify `tests/conftest.py` fixtures and data samples together.

### Common Patterns
- Bidirectional conversion between context list ↔ CSV string (semicolon-delimited)
- Approval filter (`approved`) is handled conservatively (`only_approved=True` by default)

## Dependencies

### Internal
- `src/llm/deepeval_model.py`
- `src/settings.py`

### External
- `deepeval` Synthesizer / Dataset
- `pandas`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
