<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# src

## Purpose
The core Python package for the project. Contains domain logic for Loop 1/2/3 and LLM/observability integration code. The `scripts/` directory calls these modules for execution.

## Key Files
| File | Description |
|------|-------------|
| `settings.py` | Central configuration based on environment variables (`Settings`, `get_settings`) |
| `__init__.py` | Package root initialization file |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `llm/` | OpenRouter/DeepEval model adapters (see `llm/AGENTS.md`) |
| `loop1_dataset/` | Synthetic → Review → Golden data loop (see `loop1_dataset/AGENTS.md`) |
| `loop2_evaluation/` | Evaluation / monitoring / prompt optimization loop (see `loop2_evaluation/AGENTS.md`) |
| `loop3_remediation/` | Improvement recommendation agent loop (see `loop3_remediation/AGENTS.md`) |
| `observability/` | Langfuse client/helpers (see `observability/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Keep execution entry point code in `scripts/`; maintain this directory as reusable functions/classes only.
- Add new features within Loop boundaries (1/2/3) whenever possible; connect cross-loop logic through `settings.py` and shared helpers.
- Do not modify generated directories (`__pycache__`, `*.egg-info`).

### Testing Requirements
- Run Loop-specific tests first.
  - Loop1 changes: `tests/test_synthesizer.py`, `tests/test_loop1_synthesizer_generation.py`
  - Loop2 changes: `tests/test_custom_metrics.py`, `tests/test_langfuse_sampling.py`, `tests/test_prompt_optimizer.py`, `tests/test_golden_sampling.py`
  - Loop3 changes: `tests/test_remediation_agent.py`

### Common Patterns
- Access settings via the `get_settings()` singleton.
- Maintain type hints (`from __future__ import annotations`) and explicit function interfaces.
- Prefer safe handling of exceptions/missing data with empty lists/default values.

## Dependencies

### Internal
- `src/llm/` is shared across Loop1/2/3.
- `src/observability/` is the common Langfuse access layer for Loop2/3.

### External
- `pydantic-settings` (configuration management)
- `deepeval` (Loop1/2 evaluation infrastructure)
- `langfuse` (observability)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
