#!/usr/bin/env python3
"""Langfuse 트레이스 시각화 데모 스크립트.

에이전트 실행을 시뮬레이션하여 Langfuse에 샘플 트레이스를 전송합니다.
실행 후 Langfuse 대시보드에서 다음을 확인할 수 있습니다:

- Traces: 에이전트 실행 추적 (coding_assistant, deep_research)
- Spans: 노드별 실행 시간 (parse_request → execute_code → verify_result)
- Generations: LLM 호출 상세 (모델, 토큰 사용량, 비용)
- Scores: 평가 점수 (NUMERIC, BOOLEAN, CATEGORICAL)

사용법:
    # 기본 실행 (3개 시나리오)
    python scripts/langfuse_demo.py

    # 시나리오 수 지정
    python scripts/langfuse_demo.py --count 10

    # 특정 에이전트만
    python scripts/langfuse_demo.py --agent coding_assistant

    # 드라이런 (실제 전송 없이 확인)
    python scripts/langfuse_demo.py --dry-run

필수 환경변수:
    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── .env 로드 ─────────────────────────────────────────────


def _load_dotenv() -> None:
    """프로젝트 루트의 .env 파일을 로드합니다."""
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# ── 시나리오 데이터 ───────────────────────────────────────


@dataclass
class SimulatedSpan:
    """시뮬레이션할 span 정보."""

    name: str
    duration_ms: float
    input_text: str
    output_text: str
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class SimulatedScenario:
    """시뮬레이션 시나리오."""

    agent_name: str
    session_id: str
    user_id: str
    user_input: str
    spans: list[SimulatedSpan]
    scores: dict[str, Any]  # name -> (value, data_type)
    tags: list[str]


def _make_coding_scenarios(count: int) -> list[SimulatedScenario]:
    """coding_assistant 시나리오를 생성합니다."""
    tasks = [
        ("파이썬으로 피보나치 함수 구현해줘", "def fibonacci(n):\n    ..."),
        (
            "FastAPI로 REST API 엔드포인트 만들어줘",
            "@app.get('/items')\nasync def get_items():",
        ),
        ("pandas DataFrame 정렬하는 코드 작성해줘", "df.sort_values(by='column')"),
        ("비동기 웹 크롤러 구현해줘", "async def crawl(url):"),
        (
            "SQLAlchemy ORM 모델 정의해줘",
            "class User(Base):\n    __tablename__ = 'users'",
        ),
        (
            "pytest 테스트 코드 작성해줘",
            "def test_function():\n    assert result == expected",
        ),
        (
            "Docker Compose 설정 파일 만들어줘",
            "services:\n  web:\n    image: python:3.12",
        ),
        (
            "Redis 캐시 래퍼 클래스 구현해줘",
            "class RedisCache:\n    def __init__(self, client):",
        ),
    ]

    scenarios = []
    for i in range(count):
        task_input, task_output = random.choice(tasks)
        passed = random.random() > 0.2  # 80% 통과율

        spans = [
            SimulatedSpan(
                name="parse_request",
                duration_ms=random.uniform(50, 200),
                input_text=task_input,
                output_text='{"intent": "code_generation", "language": "python"}',
                model="qwen/qwen3.5-9b",
                prompt_tokens=random.randint(80, 200),
                completion_tokens=random.randint(30, 80),
            ),
            SimulatedSpan(
                name="execute_code",
                duration_ms=random.uniform(500, 3000),
                input_text='{"intent": "code_generation"}',
                output_text=task_output,
                model="qwen/qwen3-coder",
                prompt_tokens=random.randint(200, 800),
                completion_tokens=random.randint(100, 500),
            ),
            SimulatedSpan(
                name="verify_result",
                duration_ms=random.uniform(200, 1000),
                input_text=task_output,
                output_text=f'{{"passed": {str(passed).lower()}, "issues": []}}',
                model="qwen/qwen3.5-9b",
                prompt_tokens=random.randint(150, 400),
                completion_tokens=random.randint(50, 150),
            ),
        ]

        # 가끔 에러 시나리오 추가
        if not passed and random.random() > 0.5:
            spans[1].error = "SyntaxError: unexpected EOF"

        scores = {
            "quality.passed": (passed, "BOOLEAN"),
            "quality.code_correctness": (
                random.uniform(0.6, 1.0) if passed else random.uniform(0.2, 0.6),
                "NUMERIC",
            ),
            "quality.response_time_ms": (sum(s.duration_ms for s in spans), "NUMERIC"),
            "quality.risk_level": (
                random.choice(["low", "medium", "high"]),
                "CATEGORICAL",
            ),
        }

        scenarios.append(
            SimulatedScenario(
                agent_name="coding_assistant",
                session_id=f"demo-session-{uuid.uuid4().hex[:8]}",
                user_id=f"demo-user-{random.randint(1, 5)}",
                user_input=task_input,
                spans=spans,
                scores=scores,
                tags=["demo", "coding_assistant", "env:local"],
            )
        )

    return scenarios


def _make_research_scenarios(count: int) -> list[SimulatedScenario]:
    """deep_research 시나리오를 생성합니다."""
    topics = [
        ("LangGraph와 CrewAI 비교 분석해줘", "LangGraph vs CrewAI 비교 보고서..."),
        ("RAG 아키텍처 최신 트렌드 조사해줘", "2025년 RAG 아키텍처 트렌드..."),
        (
            "멀티에이전트 시스템 설계 패턴 정리해줘",
            "멀티에이전트 시스템 설계 패턴 개요...",
        ),
        ("LLM 평가 방법론 비교해줘", "LLM 평가 프레임워크 비교 분석..."),
    ]

    scenarios = []
    for i in range(count):
        topic_input, topic_output = random.choice(topics)

        spans = [
            SimulatedSpan(
                name="clarify_with_user",
                duration_ms=random.uniform(100, 300),
                input_text=topic_input,
                output_text='{"clarified_query": "...", "scope": "comprehensive"}',
                model="qwen/qwen3.5-9b",
                prompt_tokens=random.randint(100, 250),
                completion_tokens=random.randint(50, 120),
            ),
            SimulatedSpan(
                name="write_research_brief",
                duration_ms=random.uniform(200, 500),
                input_text='{"clarified_query": "..."}',
                output_text='{"brief": "연구 계획서...", "subtopics": [...]}',
                model="deepseek/deepseek-v3.2",
                prompt_tokens=random.randint(300, 600),
                completion_tokens=random.randint(200, 400),
            ),
            SimulatedSpan(
                name="research_supervisor",
                duration_ms=random.uniform(2000, 8000),
                input_text='{"brief": "연구 계획서..."}',
                output_text='{"findings": [...], "sources": 12}',
                model="deepseek/deepseek-v3.2",
                prompt_tokens=random.randint(1000, 3000),
                completion_tokens=random.randint(500, 1500),
            ),
            SimulatedSpan(
                name="final_report_generation",
                duration_ms=random.uniform(1000, 4000),
                input_text='{"findings": [...]}',
                output_text=topic_output,
                model="deepseek/deepseek-v3.2",
                prompt_tokens=random.randint(2000, 5000),
                completion_tokens=random.randint(1000, 3000),
            ),
        ]

        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in spans)

        scores = {
            "quality.completeness": (random.uniform(0.7, 1.0), "NUMERIC"),
            "quality.source_quality": (random.uniform(0.6, 0.95), "NUMERIC"),
            "quality.response_time_ms": (sum(s.duration_ms for s in spans), "NUMERIC"),
            "quality.depth_level": (
                random.choice(["shallow", "moderate", "deep"]),
                "CATEGORICAL",
            ),
            "quality.total_tokens": (float(total_tokens), "NUMERIC"),
        }

        scenarios.append(
            SimulatedScenario(
                agent_name="deep_research",
                session_id=f"demo-session-{uuid.uuid4().hex[:8]}",
                user_id=f"demo-user-{random.randint(1, 5)}",
                user_input=topic_input,
                spans=spans,
                scores=scores,
                tags=["demo", "deep_research", "env:local"],
            )
        )

    return scenarios


# ── Langfuse 전송 ─────────────────────────────────────────


def send_trace(scenario: SimulatedScenario, *, langfuse_client: Any) -> str:
    """시나리오를 Langfuse에 트레이스로 전송합니다.

    Args:
        scenario: 시뮬레이션 시나리오
        langfuse_client: Langfuse 클라이언트 인스턴스

    Returns:
        str: 생성된 trace ID
    """
    # 트레이스 생성
    trace = langfuse_client.trace(
        name=f"{scenario.agent_name}:demo",
        user_id=scenario.user_id,
        session_id=scenario.session_id,
        input={"messages": [{"role": "user", "content": scenario.user_input}]},
        tags=scenario.tags,
        metadata={
            "env": "demo",
            "service_name": "youngs75-a2a",
            "app_version": "0.1.0",
            "demo": True,
        },
    )

    # 각 span 생성
    for span_data in scenario.spans:
        span_kwargs: dict[str, Any] = {
            "name": span_data.name,
            "input": span_data.input_text,
            "output": span_data.output_text,
            "start_time": _offset_time(-(span_data.duration_ms / 1000)),
        }

        if span_data.error:
            span_kwargs["level"] = "ERROR"
            span_kwargs["status_message"] = span_data.error

        if span_data.model:
            # LLM 호출은 generation으로 기록 (모델 정보, 토큰 사용량 포함)
            generation = trace.generation(
                name=f"{span_data.name}:llm",
                model=span_data.model,
                input=span_data.input_text,
                output=span_data.output_text,
                usage={
                    "input": span_data.prompt_tokens,
                    "output": span_data.completion_tokens,
                    "unit": "TOKENS",
                },
                metadata={"node": span_data.name},
            )
            # generation도 별도로 end() 호출
            generation.end()

        # span 자체도 기록
        span = trace.span(**span_kwargs)
        span.end()

    # 최종 출력 설정
    last_span = scenario.spans[-1] if scenario.spans else None
    trace.update(
        output={"response": last_span.output_text if last_span else ""},
    )

    # 스코어 기록
    for score_name, (score_value, data_type) in scenario.scores.items():
        score_kwargs: dict[str, Any] = {
            "name": score_name,
            "data_type": data_type,
        }

        if data_type == "BOOLEAN":
            score_kwargs["value"] = 1.0 if bool(score_value) else 0.0
        elif data_type == "CATEGORICAL":
            score_kwargs["value"] = str(score_value)
        else:
            score_kwargs["value"] = float(score_value)

        trace.score(**score_kwargs)

    return trace.id


def _offset_time(seconds: float) -> Any:
    """현재 시각에서 offset만큼 이동한 datetime을 반환합니다."""
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


# ── 드라이런 출력 ─────────────────────────────────────────


def print_scenario_summary(scenario: SimulatedScenario, index: int) -> None:
    """시나리오 정보를 콘솔에 출력합니다."""
    total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in scenario.spans)
    total_duration = sum(s.duration_ms for s in scenario.spans)
    has_error = any(s.error for s in scenario.spans)

    print(f"\n  [{index + 1}] {scenario.agent_name}")
    print(f"      입력: {scenario.user_input[:60]}...")
    print(f"      세션: {scenario.session_id}")
    print(f"      사용자: {scenario.user_id}")
    print(f"      Span 수: {len(scenario.spans)}")
    print(f"      총 토큰: {total_tokens:,}")
    print(f"      총 소요: {total_duration:.0f}ms")
    print(f"      에러: {'있음' if has_error else '없음'}")
    print(f"      스코어: {len(scenario.scores)}개")
    for name, (value, dtype) in scenario.scores.items():
        if dtype == "NUMERIC":
            print(f"        {name}: {value:.4f} ({dtype})")
        else:
            print(f"        {name}: {value} ({dtype})")


# ── 메인 ──────────────────────────────────────────────────


def main() -> int:
    """메인 진입점."""
    parser = argparse.ArgumentParser(
        description="Langfuse 트레이스 시각화 데모",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=3,
        help="에이전트별 시나리오 수 (기본: 3)",
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=["coding_assistant", "deep_research", "all"],
        default="all",
        help="시뮬레이션할 에이전트 (기본: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 전송 없이 시나리오만 출력",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 시나리오 생성
    scenarios: list[SimulatedScenario] = []
    if args.agent in ("coding_assistant", "all"):
        scenarios.extend(_make_coding_scenarios(args.count))
    if args.agent in ("deep_research", "all"):
        scenarios.extend(_make_research_scenarios(args.count))

    print()
    print("=" * 60)
    print("  Langfuse 트레이스 데모")
    print("=" * 60)
    print(f"  시나리오 수: {len(scenarios)}개")

    if args.dry_run:
        print("  모드: 드라이런 (실제 전송 없음)")
        for i, scenario in enumerate(scenarios):
            print_scenario_summary(scenario, i)
        print()
        return 0

    # Langfuse 클라이언트 초기화
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3100")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        print()
        print(
            "  [오류] LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY가 설정되지 않았습니다."
        )
        print("  먼저 scripts/setup_langfuse.py를 실행하여 설정을 확인하세요.")
        return 1

    try:
        from langfuse import Langfuse

        lf = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
    except ImportError:
        print("\n  [오류] langfuse 패키지가 설치되지 않았습니다.")
        print("  설치: pip install langfuse")
        return 1
    except Exception as e:
        print(f"\n  [오류] Langfuse 클라이언트 초기화 실패: {e}")
        return 1

    print(f"  대상: {host}")
    print()

    # 트레이스 전송
    trace_ids: list[str] = []
    for i, scenario in enumerate(scenarios):
        try:
            trace_id = send_trace(scenario, langfuse_client=lf)
            trace_ids.append(trace_id)
            print(
                f"  [{i + 1}/{len(scenarios)}] {scenario.agent_name} "
                f"— trace_id={trace_id[:16]}..."
            )
        except Exception as e:
            print(f"  [{i + 1}/{len(scenarios)}] {scenario.agent_name} — [오류] {e}")

    # flush하여 모든 이벤트 전송 완료
    print()
    print("  버퍼 flush 중...")
    try:
        lf.flush()
        print("  flush 완료")
    except Exception as e:
        print(f"  flush 실패: {e}")

    # 결과 요약
    print()
    print("=" * 60)
    print("  전송 결과")
    print("=" * 60)
    print(f"  전송 성공: {len(trace_ids)}/{len(scenarios)}개")
    print(f"  대시보드: {host}")
    print()
    print("  대시보드에서 확인할 수 있는 항목:")
    print("    - Traces 탭: 에이전트 실행 목록 (태그: demo)")
    print("    - 개별 Trace: span/generation 타임라인")
    print("    - Scores 탭: 품질 평가 점수")
    print("    - Sessions 탭: 세션별 대화 그룹")
    print("    - Users 탭: 사용자별 사용량")
    print()

    if trace_ids:
        print("  생성된 Trace ID 목록:")
        for tid in trace_ids:
            print(f"    {host}/trace/{tid}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
