<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# llm

## Purpose
Provides the OpenRouter-based LLM client and DeepEval-compatible model adapter. Serves as the common layer that allows Loop 1/2/3 to reuse the same model configuration.

## Key Files
| File | Description |
|------|-------------|
| `openrouter.py` | OpenRouter API client creation/invocation utilities |
| `deepeval_model.py` | LLM interface adapter required by DeepEval |
| `__init__.py` | Module initialization |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Do not hardcode model names, API keys, or base URLs; inject them via `src/settings.py`.
- Keep responsibilities separate between the DeepEval adapter and the general OpenRouter client.
- Maintain clear exception messages so that upper loops can handle retries/fallbacks.

### Testing Requirements
- `.venv/bin/pytest tests/test_openrouter_model.py -q`
- Run Loop1/Loop2 smoke tests when related changes are made.

### Common Patterns
- Maintain a single gateway (OpenRouter) policy
- Control temperature/model/token limits via function arguments or settings

## Dependencies

### Internal
- `src/settings.py`

### External
- `openai` SDK (uses OpenRouter OpenAI-compatible endpoint)
- `deepeval` model interface

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
