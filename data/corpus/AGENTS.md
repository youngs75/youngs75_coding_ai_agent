<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# corpus

## Purpose
The source document repository referenced when generating synthetic Q&A in Loop1 Step1.

## Key Files
| File | Description |
|------|-------------|
| `00_sla.md` | Foundational knowledge document on SLA concepts/metrics |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Document content directly affects Golden quality; maintain factual accuracy and consistency.
- When adding/modifying files, note in the log that Step1 generation diversity and quality may be affected.
- Do not store sensitive/confidential information.

### Testing Requirements
- After changes, run Step1 or Step4 (`--skip-review`) to verify that synthetic generation works correctly.

### Common Patterns
- UTF-8 markdown documents
- Maintain one clear topic per file

## Dependencies

### Internal
- `src/loop1_dataset/synthesizer.py`

### External
- None

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
