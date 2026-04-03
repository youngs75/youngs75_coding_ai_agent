<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# observability

## Purpose
A shared module responsible for Langfuse integration (client creation, availability checks, score recording).

## Key Files
| File | Description |
|------|-------------|
| `langfuse.py` | Langfuse SDK client / helper functions |
| `__init__.py` | Module initialization |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Read credentials only from `.env` and `Settings`; never embed them directly in code.
- Maintain the `enabled()` guard pattern so the module operates safely in environments without Langfuse configured.
- When changing score naming rules (including prefixes), reflect changes in Loop2 documentation/script options.

### Testing Requirements
- Prefer Loop2 mocked tests (`tests/test_langfuse_sampling.py`) over direct network call tests.

### Common Patterns
- Return empty results / harmless no-ops on external integration failures
- Accept both dict and object formats in trace/score handling

## Dependencies

### Internal
- `src/settings.py`

### External
- `langfuse` Python SDK

## Architecture: Langfuse → Fail → Improve Loop

This module handles the Observability layer in the Day3 project's **Observation → Evaluation → Improvement** closed loop.

### Overall Data Flow

```
Production LLM Call
  │
  ▼
Langfuse Trace Recording (enrich_trace / build_langchain_config)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Path A: Existing Score-based Monitoring (Step 6)       │
│  monitor_langfuse_scores()                              │
│    1. fetch_traces() — Query traces by time + tag filter│
│    2. _extract_prefixed_scores() — Extract prefix-      │
│       matched scores                                    │
│    3. score < threshold → marked as failed              │
│    4. → Save to langfuse_failed_samples.json            │
│                                                         │
│  Path B: DeepEval Re-evaluation (batch_evaluate_langfuse│
│    1. fetch_traces() — Query traces                     │
│    2. trace_to_testcase() — Convert to LLMTestCase      │
│    3. metric.measure() — Run DeepEval metrics           │
│    4. push_scores() — Record "deepeval.*" scores to     │
│       Langfuse                                          │
│    5. → Save to langfuse_batch_results.json             │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Improve Loop                                           │
│  Step 7: Remediation Agent (failure analysis →          │
│          improvement recommendations)                   │
│  Step 8: Prompt Optimizer (failure hints → evaluation   │
│          prompt improvement)                            │
│    - Passes langfuse_failed_samples.json as             │
│      external_failure_hints to LLM for automatic        │
│      prompt improvement                                 │
└─────────────────────────────────────────────────────────┘
```

### Functions Provided by This Module and Their Roles in the Loop

| Function | Role in Loop | Call Location |
|----------|-------------|--------------|
| `enabled()` | Guard for Langfuse availability | Entry point of all Langfuse integration code |
| `client()` | Langfuse SDK singleton | `langfuse_bridge.py`, `analysis_tools.py` |
| `score_trace()` | Record evaluation results as Langfuse Scores (Push) | `langfuse_bridge.push_scores()` |
| `enrich_trace()` | Add metadata/tags to production Traces | During agent execution |
| `build_langchain_config()` | Inject Langfuse metadata into LangChain run config | Loop 3 agent |

### Execution

```bash
# Monitoring + failure extraction (Path A)
python scripts/run_pipeline.py --step 6 --lf-hours 24 --lf-threshold 0.7

# Full monitoring → improvement loop
python scripts/run_pipeline.py --from 6 --to 8

# Offline evaluation only (without Langfuse)
python scripts/run_pipeline.py --step 5
```

### Local Langfuse Setup

You can run Langfuse locally with Docker Desktop:

```bash
docker compose -f docker-compose.langfuse.yaml up -d
# → Access Langfuse dashboard at http://localhost:3000
```

Set the following in `.env` for integration:

```
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=<Langfuse Project Public Key>
LANGFUSE_SECRET_KEY=<Langfuse Project Secret Key>
LANGFUSE_TRACING_ENABLED=1
```

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
