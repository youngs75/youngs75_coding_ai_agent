<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# review

## Purpose
Stores CSVs used in the Human Review stage. Reviewers fill in `approved` and `feedback` fields, which are then incorporated in Step3.

## Key Files
| File | Description |
|------|-------------|
| `review_dataset.csv` | Default review target CSV generated in Step2 |
| `tmp_reviewed.csv` | Temporary/experimental review CSV sample |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Maintain CSV column compatibility (`id,input,expected_output,context,source_file,approved,feedback,reviewer`).
- Standardize `approved` to `true/false` style values.
- Use the Step3 script to incorporate completed reviews.

### Testing Requirements
- After Step3 (`scripts/03_import_reviewed.py`) execution, verify `data/golden/golden_dataset.json` is generated

### Common Patterns
- UTF-8-SIG CSV (Excel compatible)
- Context uses semicolon (`;`) delimited strings

## Dependencies

### Internal
- `src/loop1_dataset/csv_exporter.py`
- `src/loop1_dataset/csv_importer.py`

### External
- `pandas`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
