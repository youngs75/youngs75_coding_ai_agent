<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# synthetic

## Purpose
Stores synthetic dataset JSON generated in Step1.

## Key Files
| File | Description |
|------|-------------|
| `synthetic_dataset.json` | Step1 generation output (question/expected answer/context/source info) |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- This file is treated as a generated artifact. If structural changes are needed, modify `src/loop1_dataset/synthesizer.py` and regenerate.
- If manual editing is required, record the reason in the operations log.

### Testing Requirements
- Verify that Step2 CSV export (`scripts/02_export_for_review.py`) works correctly.

### Common Patterns
- JSON list format
- Each item contains `id`, `input`, `expected_output`, `context`, `source_file`

## Dependencies

### Internal
- `scripts/01_generate_synthetic.py`
- `src/loop1_dataset/synthesizer.py`

### External
- None

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
