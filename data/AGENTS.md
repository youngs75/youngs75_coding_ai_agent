<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# data

## Purpose
The data root directory that stores input sources (corpus) and Loop execution outputs (synthetic/review/golden/evaluation/optimization artifacts).

## Key Files
None (most files are in subdirectories).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `corpus/` | Source documents for synthetic data generation (see `corpus/AGENTS.md`) |
| `synthetic/` | Step1 output synthetic dataset (see `synthetic/AGENTS.md`) |
| `review/` | Step2/3 review CSV (see `review/AGENTS.md`) |
| `golden/` | Finalized Golden dataset (see `golden/AGENTS.md`) |
| `eval_results/` | Step5/6/7 result artifacts (see `eval_results/AGENTS.md`) |
| `prompt_optimization/` | Step8 optimization reports/results (see `prompt_optimization/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Most files except `corpus/` are execution outputs. Do not manually modify unless requested.
- For format changes, always modify the generating code first, then regenerate data files.
- If temporary experimental files are added, clean them up to avoid affecting documentation/tests.

### Testing Requirements
- Run related scripts/tests when data schema changes are made.
- Minimum verification: Check output JSON structure after Step5/Step6 execution

### Common Patterns
- JSON format: UTF-8, `ensure_ascii=False`, `indent=2`
- File paths are calculated relative to `settings.data_dir`

## Dependencies

### Internal
- `scripts/01~08`
- `src/loop1_dataset/*`, `src/loop2_evaluation/*`, `src/loop3_remediation/*`

### External
- None (data storage)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
