<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# loop2_evaluation

## Purpose
The core area of Loop 2, responsible for DeepEval metric evaluation (Step 5), Langfuse monitoring-based failure extraction (Step 6), and automatic evaluation prompt optimization (Step 8).

## Key Files
| File | Description |
|------|-------------|
| `metrics_registry.py` | RAG/Agent/Custom metric combination registry |
| `rag_metrics.py` | DeepEval RAG metric factory |
| `agent_metrics.py` | Agent quality metric factory |
| `custom_metrics.py` | Custom metric implementations |
| `prompts.py` | Original/custom evaluation prompts |
| `batch_evaluator.py` | Core execution engine for Step5/6 |
| `langfuse_bridge.py` | Langfuse trace fetch / score push bridge |
| `calibration_cases.py` | Step8 calibration case definitions |
| `prompt_optimizer.py` | Meta-prompt-based automatic improvement logic |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Do not mix responsibilities across Step 5/6/8.
  - Step5: Golden offline evaluation
  - Step6: Langfuse score-based monitoring / failure extraction
  - Step8: Evaluation prompt improvement
- When changing score thresholds/sampling options, update CLI scripts (`scripts/05_run_eval.py`, `scripts/06_batch_eval_langfuse.py`, `scripts/run_pipeline.py`) and documentation simultaneously.
- Prompt constant names must match mapping keys in `prompt_optimizer.py`.

### Testing Requirements
- `.venv/bin/pytest tests/test_custom_metrics.py tests/test_langfuse_sampling.py tests/test_prompt_optimizer.py tests/test_calibration_cases.py tests/test_golden_sampling.py -q`
- When metric interfaces change, also verify `eval/test_agent_eval.py`.

### Common Patterns
- Ensure reproducibility with deterministic sampling (`sample_seed`)
- Metrics with insufficient required inputs are separated into `skipped_metrics`
- Langfuse failure sample JSON is reused as Step8 hint input

## Dependencies

### Internal
- `src/llm/` (DeepEval model)
- `src/observability/langfuse.py`
- `src/settings.py`

### External
- `deepeval`
- `langfuse`
- `pydantic`/`json` standardization handling

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
