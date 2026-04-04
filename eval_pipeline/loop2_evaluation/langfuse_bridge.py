"""Langfuse ↔ DeepEval 외부 평가 파이프라인 브릿지 모듈 (Loop 2).

Langfuse에 저장된 프로덕션 trace 데이터를 DeepEval 평가 시스템으로 연결하는
**External Evaluation Pipeline** 패턴을 구현합니다.
Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

External Evaluation Pipeline이란?
    프로덕션 LLM 트레이스를 오프라인으로 처리하여 커스텀 평가를 수행하고,
    결과를 다시 Langfuse에 저장하는 아키텍처입니다.
    Langfuse 공식 문서가 권장하는 3단계 패턴을 따릅니다:

    1단계: Fetch — Langfuse SDK로 트레이스 조회 (시간 범위 + 태그 필터)
    2단계: Evaluate — DeepEval 메트릭으로 품질 평가 (GEval, RAG 메트릭 등)
    3단계: Push — 평가 결과를 Langfuse Score로 다시 기록

전체 아키텍처:

    ┌──────────────────────────────────────────────────────────────┐
    │                    Langfuse (관측성 플랫폼)                    │
    │  ┌─────────────┐                        ┌────────────────┐  │
    │  │  Traces DB   │──── 1. fetch ─────────▶│  Score 대시보드  │  │
    │  │  (입출력,    │     fetch_traces()     │  (평가 결과     │  │
    │  │   토큰,      │                        │   시각화)       │  │
    │  │   레이턴시)   │                        │                │  │
    │  └─────────────┘                        └────────────────┘  │
    │         │                                       ▲            │
    └─────────┼───────────────────────────────────────┼────────────┘
              │                                       │
              ▼                                       │
    ┌─────────────────┐                    ┌─────────────────────┐
    │ trace_to_testcase│                    │    push_scores()    │
    │  Langfuse trace  │                    │  DeepEval 결과 →    │
    │  → LLMTestCase   │                    │  Langfuse Score     │
    └────────┬────────┘                    └──────────▲──────────┘
             │                                        │
             ▼                                        │
    ┌──────────────────────────────────────────────────┐
    │              DeepEval 평가 엔진                    │
    │  metric.measure(test_case) → score + reason      │
    │  GEval (CoT 기반), RAG 메트릭, Custom 메트릭       │
    │  Ref: https://deepeval.com/docs/metrics-llm-evals │
    └──────────────────────────────────────────────────┘

    Ref (전체 파이프라인): https://langfuse.com/docs/scores/external-evaluation-pipelines
    Ref (스코어 기록): https://langfuse.com/docs/scores/custom
    Ref (트레이스 조회): https://langfuse.com/docs/query-traces

Langfuse ↔ DeepEval 연동 방식 (공식 문서 기반):
    Langfuse 공식 문서에서 DeepEval을 외부 평가 도구로 사용하는 예시를 제공합니다:

    1. Langfuse trace에서 input/output을 추출
    2. DeepEval의 LLMTestCase(input=trace.input, actual_output=trace.output)로 변환
    3. GEval 또는 사전정의 메트릭의 measure()로 평가 실행
    4. langfuse.create_score(trace_id, name, value, comment)로 결과 기록
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    DeepEval의 GEval은 Chain-of-Thought(CoT) 프롬프팅으로 평가 기준을 자동 생성하며,
    metric.measure(test_case) 호출 후 metric.score (0~1)와 metric.reason을 반환합니다.
    Ref: https://deepeval.com/docs/metrics-llm-evals

Score 네이밍 규칙:
    - 접두사 "deepeval."을 사용하여 기존 스코어와 구분
    - 예: deepeval.faithfulness, deepeval.answer_relevancy
    - Langfuse 대시보드에서 "deepeval.*" 필터로 Day3 평가 스코어만 조회 가능
    Ref: https://langfuse.com/docs/scores/custom

Score 데이터 타입:
    push_scores()는 세 가지 Langfuse 스코어 타입을 모두 지원합니다:
    - NUMERIC (float): 연속 수치 → 시계열 추적, 평균/분포 분석
    - BOOLEAN (bool → 1.0/0.0): 이진 판정 → 통과/실패 필터링
    - CATEGORICAL (str): 범주형 → 대시보드 그룹핑/필터링
    Ref: https://langfuse.com/docs/scores/custom

    Langfuse의 Score는 비동기 연결(Asynchronous Linking)을 지원합니다:
    "If a score is ingested manually using a trace_id to link the score to a trace,
     it is not necessary to wait until the trace has been created."
    즉, trace가 아직 ingestion 중이어도 score를 먼저 보낼 수 있습니다.

    중복 방지를 위해 score_id를 "<trace_id>-<score_name>" 형식으로 설정하면
    동일 trace에 같은 이름의 스코어를 업데이트(upsert)할 수 있습니다.
    Ref: https://langfuse.com/docs/scores/custom

프로덕션 배포 패턴:
    Langfuse 공식 문서가 권장하는 배치 처리 방식:
    - 배치 크기 10개 단위로 메모리 관리 + 체크포인트로 재시작 가능
    - cron 스케줄링 (예: 매일 새벽 5시) 또는 웹훅 트리거로 실행
    - 태그 기반 필터링으로 평가 대상 트레이스를 선별
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

사용 예시 (전체 External Evaluation Pipeline):
    from youngs75_a2a.eval_pipeline.loop2_evaluation.langfuse_bridge import (
        fetch_traces, trace_to_testcase, push_scores, ScoreEntry,
    )
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    # ── 1단계: Langfuse에서 트레이스 가져오기 (태그 기반 필터링) ──
    # Ref: https://langfuse.com/docs/query-traces
    traces = fetch_traces(tags=["env:prod"], from_hours_ago=24)

    # ── 2단계: DeepEval로 평가 ──
    # Ref: https://deepeval.com/docs/metrics-llm-evals
    for trace in traces:
        tc = trace_to_testcase(trace)
        if tc:
            # GEval: CoT 기반 커스텀 평가 (Langfuse 공식 문서 예시)
            metric = GEval(
                name="Correctness",
                criteria="답변이 질문에 정확하게 응답하는지 평가",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            )
            metric.measure(tc)  # → metric.score, metric.reason

            # ── 3단계: 결과를 Langfuse Score로 푸시 ──
            # Ref: https://langfuse.com/docs/scores/custom
            push_scores(trace.id, {"correctness": metric.score})

            # 다중 타입 스코어
            push_scores(trace.id, {
                "all_passed": ScoreEntry(value=True, data_type="BOOLEAN"),
                "risk_level": ScoreEntry(value="low", data_type="CATEGORICAL"),
            })
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from deepeval.test_case import LLMTestCase

from youngs75_a2a.eval_pipeline.observability.langfuse import (
    client,
    enabled,
    score_trace,
)
from youngs75_a2a.eval_pipeline.settings import get_settings


@dataclass
class ScoreEntry:
    """개별 스코어 엔트리 (다양한 데이터 타입 지원).

    push_scores()에서 NUMERIC 외에 BOOLEAN, CATEGORICAL 스코어를
    전달하기 위한 구조체입니다.

    Langfuse는 세 가지 스코어 데이터 타입을 지원합니다:
    Ref: https://langfuse.com/docs/scores/custom

    Attributes:
        value: 스코어 값
            - NUMERIC: float (예: 0.85) → 시계열 추적, 평균/분포 분석에 적합
            - BOOLEAN: bool (예: True) → 통과/실패 이진 판정에 적합
            - CATEGORICAL: str (예: "low", "medium", "high") → 그룹핑/필터링에 적합
        data_type: 스코어 데이터 타입 (기본: "NUMERIC")
        comment: 선택적 코멘트.
            Langfuse 대시보드에서 스코어 옆에 표시되어 맥락을 제공합니다.
            예: "Faithfulness가 기준값(0.7) 미만으로 실패"

    사용 예시:
        ScoreEntry(value=0.85)                                    # NUMERIC
        ScoreEntry(value=True, data_type="BOOLEAN")               # BOOLEAN
        ScoreEntry(value="low", data_type="CATEGORICAL",
                   comment="낮은 위험")                            # CATEGORICAL
    """

    value: float | str | bool
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "NUMERIC"
    comment: str | None = None


def _has_score_prefix(trace: Any, prefix: str) -> bool:
    score_name_prefix = f"{prefix}."
    scores = getattr(trace, "scores", None)
    if not scores:
        return False
    for score in scores:
        score_name = getattr(score, "name", "")
        if isinstance(score_name, str) and score_name.startswith(score_name_prefix):
            return True
    return False


def _deterministic_sample(
    traces: list[Any],
    *,
    sample_ratio: float | None,
    max_sample_size: int | None,
    sample_seed: int,
) -> list[Any]:
    if not traces:
        return []

    if sample_ratio is None and max_sample_size is None:
        return traces

    ratio = 1.0 if sample_ratio is None else max(0.0, min(1.0, float(sample_ratio)))
    target_size = int(len(traces) * ratio)
    if ratio > 0 and target_size == 0:
        target_size = 1
    if max_sample_size is not None:
        target_size = min(target_size, max(0, int(max_sample_size)))

    if target_size <= 0:
        return []
    if target_size >= len(traces):
        return traces

    ranked: list[tuple[int, Any]] = []
    for trace in traces:
        trace_id = str(getattr(trace, "id", ""))
        key = f"{sample_seed}:{trace_id}".encode()
        rank = int(hashlib.sha256(key).hexdigest()[:16], 16)
        ranked.append((rank, trace))

    ranked.sort(key=lambda item: item[0])
    return [trace for _, trace in ranked[:target_size]]


def fetch_traces(
    *,
    tags: list[str] | None = None,
    from_hours_ago: int = 24,
    limit: int = 200,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
    sample_seed: int = 42,
    exclude_scored_prefix: str | None = None,
) -> list[Any]:
    """External Evaluation Pipeline 1단계: Langfuse에서 프로덕션 trace를 가져옵니다.

    Langfuse Python SDK의 fetch_traces() API를 사용하여
    시간 범위와 태그로 필터링한 트레이스를 조회합니다.
    이 함수는 Langfuse 공식 문서의 External Evaluation Pipeline에서
    "Fetch traces" 단계에 해당합니다.
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    Langfuse 트레이스 조회 API:
        Langfuse Python SDK는 두 가지 트레이스 조회 방식을 제공합니다:
        1. lf.fetch_traces() — 이 함수에서 사용하는 방식. 페이지네이션된 결과 반환
        2. lf.api.trace.list() — REST API 미러. 더 세밀한 필터 지원
        Ref: https://langfuse.com/docs/query-traces

        주요 필터 파라미터:
        - tags: 태그 기반 필터링 (enrich_trace/build_langchain_config에서 설정한 태그와 매칭)
        - from_timestamp: 시작 시각 (UTC)
        - limit: 페이지당 최대 건수
        추가로 user_id, session_id, name 등으로도 필터 가능
        Ref: https://langfuse.com/docs/tracing-features/tags

        참고: "new data is typically available for querying within 15-30 seconds"
        트레이스 생성 직후에는 조회되지 않을 수 있습니다.
        Ref: https://langfuse.com/docs/query-traces

    Langfuse가 설정되지 않은 경우 빈 리스트를 반환합니다.

    Args:
        tags: 필터링할 태그 목록 (예: ["env:prod", "version:0.1.0"])
            enrich_trace()의 tags나 build_langchain_config()의 tags로 설정한 값과 매칭
        from_hours_ago: 현재 시간으로부터 몇 시간 전까지 조회할지 (기본 24시간)
            프로덕션 배포 시 cron 주기와 맞추면 누락 방지
            Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines
        limit: 최대 조회 건수 (기본 200)
            메모리 관리를 위해 배치 크기 10~200 사이 권장
        sample_ratio: 조회된 trace 중 평가할 비율 (0.0~1.0).
            None이면 비율 샘플링을 적용하지 않습니다.
        max_sample_size: 샘플 최대 건수 상한. None이면 상한 없음.
        sample_seed: deterministic 샘플링 시드 (기본 42).
        exclude_scored_prefix: 해당 접두사 스코어가 이미 있는 trace는 제외.
            예: "deepeval" 지정 시 deepeval.* 스코어가 있는 trace를 스킵.

    Returns:
        Langfuse trace 객체 리스트.
        각 trace 객체는 .id, .input, .output, .metadata 등의 속성을 가짐
    """
    settings = get_settings()
    if not enabled(settings):
        return []

    lf = client()
    from_ts = datetime.now(tz=UTC) - timedelta(hours=from_hours_ago)

    # Langfuse SDK v3: lf.api.trace.list() + lf.api.trace.get() 사용
    # v2의 lf.fetch_traces()는 v3에서 제거됨
    response = lf.api.trace.list(
        tags=tags,
        from_timestamp=from_ts,
        limit=limit,
    )
    trace_summaries = response.data if hasattr(response, "data") else []

    # list() 결과는 scores를 ID만 포함하므로, get()으로 상세 조회
    traces = []
    for summary in trace_summaries:
        try:
            detailed = lf.api.trace.get(summary.id)
            traces.append(detailed)
        except Exception:
            traces.append(summary)

    if exclude_scored_prefix:
        traces = [
            trace
            for trace in traces
            if not _has_score_prefix(trace, exclude_scored_prefix)
        ]

    return _deterministic_sample(
        traces,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
        sample_seed=sample_seed,
    )


def trace_to_testcase(trace: Any) -> LLMTestCase | None:
    """External Evaluation Pipeline 1→2단계 변환: Langfuse trace → DeepEval LLMTestCase.

    Langfuse trace 객체에서 input(질문), output(답변), context(컨텍스트)를
    추출하여 DeepEval이 평가할 수 있는 LLMTestCase로 변환합니다.
    이 변환은 Langfuse 공식 문서의 External Evaluation Pipeline에서
    "trace → evaluation input" 매핑에 해당합니다.
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    Langfuse 공식 문서의 DeepEval 연동 예시:
        Langfuse 문서에서는 다음과 같이 trace를 LLMTestCase로 변환합니다:
        ```
        test_case = LLMTestCase(
            input=trace.input["args"],     # trace의 입력
            actual_output=trace.output,     # trace의 출력 (LLM 응답)
        )
        ```
        Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

        이 함수는 위 패턴을 확장하여 다양한 trace 구조를 처리합니다:
        - input이 dict일 때 "query", "input" 키를 순서대로 탐색
        - output이 dict일 때 "answer", "output" 키를 순서대로 탐색
        - metadata에서 RAG 평가용 context/retrieval_context도 추출

    변환 매핑:
        trace.input  (dict: "query"/"input" 키) → LLMTestCase.input
        trace.output (dict: "answer"/"output" 키) → LLMTestCase.actual_output
        trace.metadata["context"]           → LLMTestCase.context
        trace.metadata["retrieval_context"] → LLMTestCase.retrieval_context

    DeepEval LLMTestCase 필드 설명:
        - input: 사용자 질문 (필수) — 모든 메트릭에서 사용
        - actual_output: LLM의 실제 응답 (필수) — 평가 대상
        - context: 정답에 필요한 참조 컨텍스트 (선택) — ContextualPrecision/Recall에서 사용
        - retrieval_context: 검색된 컨텍스트 (선택) — Faithfulness에서 사용
        Ref: https://deepeval.com/docs/metrics-llm-evals

    metadata의 context/retrieval_context는 enrich_trace()의 metadata로
    설정하거나, 에이전트 실행 시 직접 트레이스에 기록할 수 있습니다.
    Ref: https://langfuse.com/docs/tracing-features/metadata

    input 또는 output이 없으면 None을 반환합니다 (평가 불가).

    Args:
        trace: Langfuse trace 객체 (fetch_traces()가 반환하는 객체)
            주요 속성: .id, .input (dict|str), .output (dict|str), .metadata (dict)

    Returns:
        LLMTestCase 또는 None (input/output 누락 시 변환 불가)
    """
    try:
        input_text = ""
        actual_output = ""
        context = []
        retrieval_context = []

        # input 추출: dict인 경우 query/input 키를 순서대로 탐색
        if hasattr(trace, "input") and trace.input:
            if isinstance(trace.input, dict):
                input_text = (
                    trace.input.get("query", "")
                    or trace.input.get("input", "")
                    or str(trace.input)
                )
            else:
                input_text = str(trace.input)

        # output 추출: dict인 경우 answer/output 키를 순서대로 탐색
        if hasattr(trace, "output") and trace.output:
            if isinstance(trace.output, dict):
                actual_output = (
                    trace.output.get("answer", "")
                    or trace.output.get("output", "")
                    or str(trace.output)
                )
            else:
                actual_output = str(trace.output)

        # metadata에서 context/retrieval_context 추출
        if hasattr(trace, "metadata") and isinstance(trace.metadata, dict):
            ctx = trace.metadata.get("context", [])
            if isinstance(ctx, list):
                context = ctx
            ret_ctx = trace.metadata.get("retrieval_context", [])
            if isinstance(ret_ctx, list):
                retrieval_context = ret_ctx

        # input 또는 output이 없으면 평가 불가
        if not input_text or not actual_output:
            return None

        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context=context if context else None,
            retrieval_context=retrieval_context if retrieval_context else None,
        )
    except Exception as exc:
        trace_id = getattr(trace, "id", "unknown")
        print(f"[WARN] trace_to_testcase failed for trace '{trace_id}': {exc}")
        return None


def push_scores(
    trace_id: str,
    results: dict[str, float | ScoreEntry],
    *,
    prefix: str = "deepeval",
) -> None:
    """External Evaluation Pipeline 3단계: DeepEval 평가 결과를 Langfuse Score로 푸시합니다.

    Langfuse 공식 문서의 External Evaluation Pipeline에서 마지막 "Push scores" 단계입니다.
    각 메트릭 결과를 "{prefix}.{metric_name}" 형식의 스코어로 Langfuse에 기록합니다.
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    Langfuse 공식 문서의 DeepEval 연동 예시:
        Langfuse 문서에서는 DeepEval 평가 후 다음과 같이 스코어를 기록합니다:
        ```
        jscore = joyfulness_score(trace)  # DeepEval metric.measure() 결과
        langfuse.create_score(
            trace_id=trace.id,
            name="joyfulness",
            value=jscore["score"],         # NUMERIC (0~1)
            comment=jscore["reason"],      # DeepEval의 평가 이유
        )
        ```
        Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

        이 함수는 위 패턴을 자동화하여 여러 메트릭 결과를 한 번에 푸시하고,
        NUMERIC 외에 BOOLEAN, CATEGORICAL 타입도 지원합니다.

    내부적으로 src.observability.langfuse.score_trace()를 호출하여
    NUMERIC, BOOLEAN, CATEGORICAL 세 가지 데이터 타입을 모두 지원합니다.
    Ref: https://langfuse.com/docs/scores/custom

    하위호환성:
        기존 NUMERIC 전용 사용법과 100% 호환됩니다.
        dict[str, float]로 전달하면 기존과 동일하게 NUMERIC으로 처리됩니다.
        batch_evaluator.py의 push_scores(trace_id, scores) 호출이
        수정 없이 그대로 동작합니다.

    기존 사용법 (NUMERIC만 — Langfuse 공식 예시와 동일):
        push_scores(tid, {"faithfulness": 0.85})  # 기존과 동일하게 동작

    새로운 다중 타입 사용법:
        push_scores(tid, {
            "faithfulness": 0.85,                                      # float → NUMERIC
            "all_passed": ScoreEntry(value=True, data_type="BOOLEAN"), # BOOLEAN
            "risk_level": ScoreEntry(value="low", data_type="CATEGORICAL"),
        })

    Args:
        trace_id: Langfuse trace ID (fetch_traces()에서 trace.id로 획득)
        results: 메트릭명 → 점수(float) 또는 ScoreEntry 딕셔너리.
            DeepEval metric.measure() 실행 후 {metric_name: metric.score}로 구성
        prefix: 스코어 이름 접두사 (기본: "deepeval").
            접두사 기반 네이밍으로 Langfuse 대시보드에서 스코어 그룹을 필터링:
            예) "deepeval.*" → Day3 평가 스코어만 조회
            Ref: https://langfuse.com/docs/scores/custom
    """
    settings = get_settings()
    if not enabled(settings):
        return

    for metric_name, entry in results.items():
        # 접두사 네이밍: "deepeval.faithfulness", "deepeval.all_passed" 등
        # Langfuse 대시보드에서 접두사로 필터링 가능
        score_name = f"{prefix}.{metric_name}"

        if isinstance(entry, ScoreEntry):
            # ScoreEntry: 사용자가 data_type과 comment을 명시적으로 지정
            # 예) ScoreEntry(value=True, data_type="BOOLEAN", comment="모두 통과")
            score_trace(
                trace_id,
                name=score_name,
                value=entry.value,
                data_type=entry.data_type,
                comment=entry.comment,
            )
        else:
            # 하위호환 경로: float/int 값은 기존과 동일하게 NUMERIC으로 처리
            # batch_evaluator.py의 push_scores(trace_id, scores) 호출이
            # 수정 없이 이 경로를 타게 됨
            score_trace(
                trace_id,
                name=score_name,
                value=entry,
                data_type="NUMERIC",
            )
