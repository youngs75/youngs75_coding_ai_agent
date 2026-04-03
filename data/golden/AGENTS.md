<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# golden

## Purpose
The finalized Golden Dataset repository. Used as input for Loop2 evaluation (Step5) and CI gates (`eval/test_agent_eval.py`).

## Key Files
| File | Description |
|------|-------------|
| `golden_dataset.json` | Final approved/augmented Golden dataset |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Prefer regenerating through Loop1 (Step3/4) over manual editing.
- Maintain schema consistency, as missing fields will cause Step5 metric skips/failures.
- Record sample count and key field changes before and after modifications in the execution log.

### Testing Requirements
- Run Step5 sampling evaluation to verify JSON structure and evaluation flow are maintained.
- Also check whether CI gate tests (`eval/test_agent_eval.py`) are affected.

### Common Patterns
- JSON array format
- Evaluation primarily uses `expected_output`, `context`, `source_file`

## Dependencies

### Internal
- `src/loop1_dataset/golden_builder.py`
- `src/loop2_evaluation/batch_evaluator.py`
- `eval/test_agent_eval.py`

### External
- None

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
