<!-- Parent: ../AGENTS.md -->

# memory

## Purpose
CoALA 논문 기반 4종 메모리 시스템. 에이전트의 장기 기억(시맨틱, 에피소딕, 절차적)과 작업 메모리를 관리한다.

## Key Files
| File | Description |
|------|-------------|
| `store.py` | `MemoryStore` — 메모리 CRUD + 벡터 검색 |
| `schemas.py` | `MemoryItem`, `MemoryType` 열거형 |
| `search.py` | 시맨틱 검색 (임베딩 기반 유사도) |
| `semantic_loader.py` | `.ai/memory/semantic.jsonl` 로더 |
| `state.py` | LangGraph 상태에 메모리 통합 |

## For AI Agents
- 메모리 타입: `SEMANTIC`, `EPISODIC`, `PROCEDURAL`, `WORKING`
- `MemoryStore`는 에이전트 초기화 시 주입되며, 노드에서 `self._memory_store`로 접근
- 시맨틱 메모리는 `.ai/memory/semantic.jsonl`에 영속화
