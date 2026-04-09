<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# scripts

## Purpose
Provides step-by-step execution entry points. Each script calls `src/` modules to execute or orchestrate Loop 1/2/3 workflows.

## Key Files
| File | Description |
|------|-------------|
| `run_pipeline.py` | Step 1–8 unified execution orchestrator |
| `01_generate_synthetic.py` | Step 1: Synthetic generation |
| `02_export_for_review.py` | Step 2: Export review CSV |
| `03_import_reviewed.py` | Step 3: Import review CSV + augmentation |
| `04_build_golden.py` | Step 4: Loop1 orchestration |
| `05_run_eval.py` | Step 5: Golden offline evaluation |
| `06_batch_eval_langfuse.py` | Step 6: Langfuse monitoring / external evaluation |
| `07_run_remediation.py` | Step 7: Remediation agent |
| `08_optimize_eval_prompts.py` | Step 8: Evaluation prompt optimization |
| `demo_full_loop.py` | End-to-end execution example for demo purposes |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Scripts should only handle argument parsing/output formatting; delegate business logic to `src/`.
- When adding new options, also reflect them in `run_pipeline.py` to maintain the unified execution path.
- Document execution examples based on `.venv/bin/python`.

### Testing Requirements
- Minimum verification for CLI changes:
  - `.venv/bin/python scripts/run_pipeline.py --help`
  - `.venv/bin/python scripts/05_run_eval.py --help`
  - Related Step unit tests

### Common Patterns
- `sys.path.insert(0, project_root)` pattern for package imports
- Summary output + artifact path guidance upon Step completion

## Dependencies

### Internal
- `src/loop1_dataset/*`
- `src/loop2_evaluation/*`
- `src/loop3_remediation/*`

### External
- `argparse`, `asyncio` (standard library)
- Loop-specific external packages are handled by `src/` modules

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
