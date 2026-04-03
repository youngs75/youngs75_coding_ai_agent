<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# prompt_optimization

## Purpose
Stores Step8 automatic prompt optimization results.

## Key Files
| File | Description |
|------|-------------|
| `report.json` | Detailed report including per-metric baseline/best/histories |
| `best_prompts.json` | Final selected prompt text per metric |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Files are Step8 execution outputs. Prefer re-running the optimization loop over manual editing.
- Before `--opt-apply`, first check fit_score improvement in `report.json`.
- Record file backup/version tagging strategy in the operations log for past result comparison.

### Testing Requirements
- After Step8 execution, verify both `report.json` and `best_prompts.json` are generated
- Re-run `tests/test_prompt_optimizer.py` if needed

### Common Patterns
- Metric keys: `response_completeness`, `citation_quality`, `safety`
- Check best fit uplift compared to baseline

## Dependencies

### Internal
- `src/loop2_evaluation/prompt_optimizer.py`
- `scripts/08_optimize_eval_prompts.py`

### External
- None

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
