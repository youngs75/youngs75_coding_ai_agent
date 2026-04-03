<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# eval_results

## Purpose
The artifact directory storing Step5/Step6/Step7 execution results. Serves as evidence data for quality trend analysis and Step8 hint inputs.

## Key Files
| File | Description |
|------|-------------|
| `eval_results.json` | Step5 Golden offline evaluation results |
| `langfuse_monitoring_snapshot.json` | Step6 monitoring aggregation snapshot |
| `langfuse_failed_samples.json` | Step6 failed sample list (Step8 input) |
| `remediation_report.json` | Step7 remediation report |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Result files are generated artifacts updated by execution. Manual editing should only be performed for analysis/debugging purposes.
- Step8 reads `langfuse_failed_samples.json` by default, so maintain file path/format stability.
- When adding new metrics, synchronize result JSON keys with documentation/dashboard fields.

### Testing Requirements
- After Step6 execution, verify `summary` and `failed_samples` structure
- Before Step8 execution, confirm the failure samples file exists and is JSON-parseable

### Common Patterns
- JSON results with timestamps
- Standardize score/threshold/fail_reason structure for reuse in subsequent steps

## Dependencies

### Internal
- `src/loop2_evaluation/batch_evaluator.py`
- `src/loop3_remediation/analysis_tools.py`
- `src/loop2_evaluation/prompt_optimizer.py`

### External
- None

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
