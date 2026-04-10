<!-- Parent: ../AGENTS.md -->

# skills

## Purpose
3-Level 스킬 시스템. 태스크 유형/프레임워크에 따라 스킬을 자동 활성화하고 프롬프트에 주입한다.

## Key Files
| File | Description |
|------|-------------|
| `registry.py` | `SkillRegistry` — 스킬 등록/조회/자동 활성화 |
| `loader.py` | YAML/JSON 스킬 정의 파일 로더 |
| `schemas.py` | 스킬 메타데이터 스키마 |

## For AI Agents
- L1: 메타데이터만 (이름, 태그), L2: 본문 로드, L3: 도구 바인딩
- `auto_activate_for_task(task_type, framework_hint)` — 태스크 기반 자동 활성화
- Orchestrator에서 `skill_registry`를 CodingAssistant에 전달해야 활성화됨
