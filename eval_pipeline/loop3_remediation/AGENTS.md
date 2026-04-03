<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# loop3_remediation

## Purpose
The Step 7 remediation agent area. Reads Loop2 evaluation results to analyze failure patterns and generates improvement recommendations as structured reports.

## Key Files
| File | Description |
|------|-------------|
| `remediation_agent.py` | DeepAgents-based Supervisor/Subagent orchestration |
| `analysis_tools.py` | Evaluation results / monitoring data retrieval tools |
| `recommendation.py` | Recommendation report Pydantic model |
| `prompts.py` | Analyzer/optimizer/recommender prompts |
| `__init__.py` | Module initialization |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Keep output schema in 1:1 alignment with the `recommendation.py` model and maintain a parsing failure fallback path.
- When changing the return format of tool functions (`analysis_tools.py`), modify prompts and parser logic together.
- When changing agent prompts, first verify JSON extraction stability (code block/text mixed content).

### Testing Requirements
- `.venv/bin/pytest tests/test_remediation_agent.py -q`
- For related changes, add tool function verification using Loop2 result file fixtures.

### Common Patterns
- Supervisor does not use tools directly; delegates to subagents
- Final results are force-structured as `RecommendationReport`

## Dependencies

### Internal
- `src/loop2_evaluation` artifacts (`data/eval_results/*`)
- `src/settings.py`

### External
- `deepagents`
- `langchain-openai`
- `langgraph`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
