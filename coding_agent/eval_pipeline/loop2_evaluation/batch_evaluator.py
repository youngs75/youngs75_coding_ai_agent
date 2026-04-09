"""배치 평가/모니터링 실행 모듈 (Loop 2 핵심).

세 가지 실행 모드를 제공합니다:

1. 오프라인 평가 (evaluate_golden_dataset):
   Golden Dataset JSON → DeepEval 메트릭 → 결과 JSON
   CI/CD 파이프라인이나 정기 평가에 사용

2. 온라인 외부평가 (batch_evaluate_langfuse):
   Langfuse Traces → DeepEval 메트릭 → Langfuse Scores + 결과 JSON
   프로덕션 환경에서 실시간 품질 모니터링에 사용
   **이 함수가 Langfuse External Evaluation Pipeline 패턴의 구현체입니다.**
   Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

3. 온라인 모니터링 스냅샷 (monitor_langfuse_scores):
   Langfuse Traces(+내장 score) → 샘플링 → 실패 추출 → 결과 JSON
   프로덕션 운영에서 "Langfuse 평가를 단일 기준"으로 사용할 때 권장

External Evaluation Pipeline (온라인 평가):
    Langfuse 공식 문서가 권장하는 외부 평가 파이프라인 아키텍처를 구현합니다.
    프로덕션 LLM 트레이스를 오프라인으로 가져와 DeepEval로 평가하고,
    결과를 Langfuse Score로 다시 기록하여 대시보드에서 모니터링합니다.
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    3단계 데이터 흐름:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1단계: Fetch (langfuse_bridge.fetch_traces)                     │
    │   Langfuse SDK로 프로덕션 트레이스 조회                            │
    │   - 시간 범위: from_hours_ago (기본 24시간)                       │
    │   - 태그 필터: tags (예: ["env:prod"])                           │
    │   - 페이지네이션: limit (기본 200)                                │
    │   Ref: https://langfuse.com/docs/query-traces                   │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2단계: Evaluate (DeepEval metric.measure)                       │
    │   각 trace를 LLMTestCase로 변환 후 DeepEval 메트릭 실행           │
    │   - trace_to_testcase(): trace.input/output → LLMTestCase       │
    │   - metric.measure(test_case): CoT 기반 LLM 평가 → score/reason │
    │   - MetricsRegistry: RAG(4) + Agent(2) + Custom(7) = 13개 메트릭 │
    │   Ref: https://deepeval.com/docs/metrics-llm-evals              │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3단계: Push (langfuse_bridge.push_scores)                       │
    │   평가 결과를 "deepeval.*" 접두사 스코어로 Langfuse에 기록          │
    │   - NUMERIC: 연속 수치 (faithfulness=0.85)                      │
    │   - BOOLEAN: 통과/실패 (all_passed=True)                        │
    │   - CATEGORICAL: 범주형 (risk_level="low")                      │
    │   → Langfuse 대시보드에서 "deepeval.*" 필터로 조회·분석 가능        │
    │   Ref: https://langfuse.com/docs/scores/custom                  │
    └─────────────────────────────────────────────────────────────────┘

    Langfuse 공식 문서의 DeepEval 연동 예시:
        ```python
        # Langfuse 공식 문서에서 제시하는 패턴
        # Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        def evaluate_trace(trace):
            metric = GEval(
                name="Correctness",
                criteria="...",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            )
            test_case = LLMTestCase(
                input=trace.input["args"],
                actual_output=trace.output,
            )
            metric.measure(test_case)
            return {"score": metric.score, "reason": metric.reason}

        result = evaluate_trace(trace)
        langfuse.create_score(
            trace_id=trace.id,
            name="correctness",
            value=result["score"],
            comment=result["reason"],
        )
        ```

    프로덕션 배포:
        Langfuse 문서가 권장하는 배포 방식:
        - 배치 크기 10개 단위 처리 → 메모리 관리 + 체크포인트 재시작
        - cron 스케줄링 (예: cron(0 5 * * ? *) → 매일 새벽 5시)
        - 또는 웹훅 트리거로 이벤트 기반 실행
        Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

오프라인 vs 온라인 아키텍처 비교:
    [오프라인 — evaluate_golden_dataset]
    Golden Dataset(JSON) → LLMTestCase → MetricsRegistry → eval_results.json
    특징: Langfuse 불필요, CI/CD에서 자동화 가능, 정적 데이터 평가

    [온라인 — batch_evaluate_langfuse (External Evaluation Pipeline)]
    Langfuse → fetch_traces → LLMTestCase → MetricsRegistry → push_scores → Langfuse
                                                              → langfuse_batch_results.json
    특징: 프로덕션 실데이터 평가, 결과가 Langfuse 대시보드에 반영됨

    [온라인 — monitor_langfuse_scores (Langfuse Native Eval Monitoring)]
    Langfuse → fetch_traces → trace.scores(prefix) → threshold fail extract
                                                  → langfuse_monitoring_snapshot.json
                                                  → langfuse_failed_samples.json
    특징: DeepEval 호출 없이 Langfuse 평가 지표 기반으로 실패 샘플 추출

사용 예시:
    from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
        evaluate_golden_dataset, batch_evaluate_langfuse, monitor_langfuse_scores,
    )

    # 오프라인 평가 (CI/CD용)
    results = evaluate_golden_dataset(metric_categories=["rag", "custom"])

    # 온라인 Langfuse 배치 평가 (External Evaluation Pipeline)
    # Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines
    results = batch_evaluate_langfuse(tags=["env:prod"], from_hours_ago=24)

    # 온라인 모니터링 스냅샷 (Langfuse score 기반)
    snapshot = monitor_langfuse_scores(tags=["env:prod"], score_prefix="eval")
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase

from coding_agent.eval_pipeline.loop2_evaluation.langfuse_bridge import (
    fetch_traces,
    push_scores,
    trace_to_testcase,
)
from coding_agent.eval_pipeline.loop2_evaluation.metrics_registry import get_registry
from coding_agent.eval_pipeline.settings import get_settings

_REQUIRED_FIELDS_BY_METRIC: dict[str, tuple[str, ...]] = {
    "faithfulness": ("retrieval_context",),
    "contextual_precision": ("retrieval_context", "expected_output"),
    "contextual_recall": ("retrieval_context", "expected_output"),
    "response_completeness_[geval]": ("expected_output",),
    "citation_quality_[geval]": ("context",),
}


def _stable_rank(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:16], 16)


def _sample_target_size(
    total: int,
    *,
    sample_ratio: float | None,
    max_sample_size: int | None,
) -> int:
    if total <= 0:
        return 0

    if sample_ratio is None:
        target = total
    else:
        ratio = max(0.0, min(1.0, float(sample_ratio)))
        target = int(total * ratio)
        if ratio > 0 and target == 0:
            target = 1

    if max_sample_size is not None:
        target = min(target, max(0, int(max_sample_size)))

    return max(0, min(total, target))


def _item_group_key(item: dict[str, Any], stratify_by: list[str]) -> tuple[str, ...]:
    key_parts: list[str] = []
    for field in stratify_by:
        value = item.get(field)
        key_parts.append(str(value) if value is not None else "__missing__")
    return tuple(key_parts)


def sample_golden_items(
    golden_items: list[dict[str, Any]],
    *,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
    sample_seed: int = 42,
    stratify_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    total = len(golden_items)
    target_size = _sample_target_size(
        total,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
    )
    if target_size >= total:
        return list(golden_items)
    if target_size <= 0:
        return []

    indexed_items = list(enumerate(golden_items))
    stratify_fields = [field for field in (stratify_by or []) if field]

    if not stratify_fields:
        ranked = sorted(
            indexed_items,
            key=lambda pair: _stable_rank(
                sample_seed,
                str(pair[1].get("id") or pair[1].get("input") or pair[0]),
            ),
        )
        selected_indexes = sorted(index for index, _ in ranked[:target_size])
        return [golden_items[index] for index in selected_indexes]

    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for index, item in indexed_items:
        group_key = _item_group_key(item, stratify_fields)
        groups.setdefault(group_key, []).append((index, item))

    group_keys = sorted(groups.keys())
    group_sizes = {key: len(groups[key]) for key in group_keys}
    group_allocations = {key: 0 for key in group_keys}

    if target_size >= len(group_keys):
        for key in group_keys:
            group_allocations[key] = 1
        remaining = target_size - len(group_keys)
    else:
        remaining = target_size

    if remaining > 0:
        residual_sizes = {
            key: max(0, group_sizes[key] - group_allocations[key]) for key in group_keys
        }
        residual_total = sum(residual_sizes.values())
        if residual_total > 0:
            raw_quota = {
                key: remaining * (residual_sizes[key] / residual_total)
                for key in group_keys
            }
            base_quota = {
                key: min(residual_sizes[key], math.floor(raw_quota[key]))
                for key in group_keys
            }
            for key in group_keys:
                group_allocations[key] += base_quota[key]
            leftover = remaining - sum(base_quota.values())
            if leftover > 0:
                order = sorted(
                    group_keys,
                    key=lambda key: (
                        -(raw_quota[key] - base_quota[key]),
                        _stable_rank(sample_seed, "|".join(key)),
                    ),
                )
                for key in order:
                    if leftover <= 0:
                        break
                    if group_allocations[key] >= group_sizes[key]:
                        continue
                    group_allocations[key] += 1
                    leftover -= 1

    selected_indices: list[int] = []
    for key in group_keys:
        take = min(group_allocations[key], group_sizes[key])
        if take <= 0:
            continue
        ranked_group = sorted(
            groups[key],
            key=lambda pair: _stable_rank(
                sample_seed,
                f"{pair[1].get('id') or pair[1].get('input') or pair[0]}|{key}",
            ),
        )
        selected_indices.extend(index for index, _ in ranked_group[:take])

    selected_indices = sorted(selected_indices)[:target_size]
    return [golden_items[index] for index in selected_indices]


def _has_required_value(test_case: LLMTestCase, field_name: str) -> bool:
    value = getattr(test_case, field_name, None)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _run_metrics_on_testcase(
    test_case: LLMTestCase,
    metrics: list,
) -> tuple[dict[str, float], list[str]]:
    """Pipeline 2단계 핵심: 단일 테스트 케이스에 대해 모든 DeepEval 메트릭을 실행합니다.

    각 메트릭의 measure() 메서드를 호출하고 결과를 수집합니다.
    DeepEval의 measure()는 내부적으로 LLM-as-a-Judge 방식으로 평가를 수행하며,
    GEval의 경우 Chain-of-Thought(CoT) 프롬프팅으로 평가 기준을 자동 생성합니다.
    Ref: https://deepeval.com/docs/metrics-llm-evals

    metric.measure(test_case) 호출 후:
        - metric.score: 0~1 범위의 평가 점수
        - metric.reason: 평가 이유 (CoT 결과)
        - metric.success: threshold 이상이면 True
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    개별 메트릭 실패 시 해당 메트릭만 0.0으로 처리하고 나머지는 계속 실행합니다.

    Args:
        test_case: 평가할 DeepEval LLMTestCase
            (trace_to_testcase() 또는 Golden Dataset에서 생성)
        metrics: 실행할 DeepEval 메트릭 인스턴스 리스트
            (MetricsRegistry에서 카테고리별로 가져옴)

    Returns:
        튜플 (scores, skipped_metrics)
            - scores: {메트릭명: 점수} 딕셔너리. 메트릭명은 snake_case로 정규화됨.
            - skipped_metrics: 필수 필드 부족으로 스킵된 메트릭명 리스트
        scores는 push_scores()에 그대로 전달하여 Langfuse Score로 기록 가능.
    """
    results = {}
    skipped_metrics: list[str] = []
    for metric in metrics:
        metric_name = getattr(metric, "__name__", metric.__class__.__name__)
        metric_key = metric_name.lower().replace(" ", "_")
        required_fields = _REQUIRED_FIELDS_BY_METRIC.get(metric_key, ())
        if required_fields and not all(
            _has_required_value(test_case, field) for field in required_fields
        ):
            skipped_metrics.append(metric_key)
            continue

        try:
            metric.measure(test_case)
            # 메트릭명을 snake_case로 정규화 (예: "Answer Relevancy" → "answer_relevancy")
            results[metric_key] = metric.score
        except Exception as exc:
            print(f"[WARN] Metric '{metric_key}' failed: {exc}")
            results[metric_key] = 0.0
    return results, skipped_metrics


def evaluate_golden_dataset(
    golden_path: Path | None = None,
    *,
    metric_categories: list[str] | None = None,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
    sample_seed: int = 42,
    stratify_by: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Golden Dataset을 DeepEval 메트릭으로 오프라인 평가합니다.

    Golden Dataset의 각 항목을 LLMTestCase로 변환하고,
    지정된 카테고리의 메트릭을 실행하여 결과를 JSON으로 저장합니다.

    Args:
        golden_path: Golden Dataset JSON 경로 (기본: data/golden/golden_dataset.json)
        metric_categories: 사용할 메트릭 카테고리 목록 (기본: ["rag", "custom"])
                          가능한 값: "rag", "agent", "custom", "all"
        sample_ratio: Golden 샘플링 비율 (0.0~1.0). None이면 비율 샘플링 미적용.
        max_sample_size: 샘플 최대 건수 상한. None이면 상한 없음.
        sample_seed: deterministic 샘플링 시드 (기본: 42)
        stratify_by: 층화 샘플링 기준 필드 목록 (예: ["source_file"])
        output_path: 결과 저장 JSON 경로 (기본: data/eval_results/eval_results.json)

    Returns:
        각 테스트 케이스별 평가 결과 리스트. 각 항목은:
            - id: Golden Dataset 항목 ID
            - input: 질문 텍스트
            - scores: {메트릭명: 점수} 딕셔너리
            - skipped_metrics: 필수 필드 부족으로 스킵된 메트릭 리스트
            - passed: 실행된 메트릭이 있고, 모든 점수가 0.5 이상이면 True
            - timestamp: 평가 시각 (ISO 형식)
    """
    settings = get_settings()
    golden_path = golden_path or (settings.data_dir / "golden" / "golden_dataset.json")
    output_path = output_path or (
        settings.data_dir / "eval_results" / "eval_results.json"
    )

    # Golden Dataset 로드
    if not golden_path.exists():
        print(f"[Eval] Golden dataset not found: {golden_path}")
        return []

    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)
    sampled_golden_data = sample_golden_items(
        golden_data,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
        sample_seed=sample_seed,
        stratify_by=stratify_by,
    )
    print(
        "[Eval] Golden sampling: "
        f"total={len(golden_data)}, selected={len(sampled_golden_data)}, "
        f"sample_ratio={sample_ratio}, max_sample_size={max_sample_size}, "
        f"stratify_by={stratify_by}"
    )

    # 메트릭 레지스트리에서 지정 카테고리의 메트릭 가져오기
    registry = get_registry()
    categories = metric_categories or ["rag", "custom"]
    metrics = []
    for cat in categories:
        metrics.extend(registry.get_metrics_by_category(cat))

    # 각 Golden 항목을 LLMTestCase로 변환하여 평가
    eval_results = []
    for item in sampled_golden_data:
        test_case = LLMTestCase(
            input=item.get("input", ""),
            # 오프라인 평가에서는 expected_output을 actual_output으로 사용
            # (실제 Agent 응답이 없으므로 Golden의 기대 답변으로 대체)
            actual_output=item.get("expected_output", ""),
            expected_output=item.get("expected_output", ""),
            context=item.get("context") if item.get("context") else None,
            retrieval_context=item.get("retrieval_context")
            if item.get("retrieval_context")
            else None,
        )

        # 메트릭 실행
        scores, skipped_metrics = _run_metrics_on_testcase(test_case, metrics)

        result = {
            "id": item.get("id", ""),
            "input": item.get("input", ""),
            "scores": scores,
            "skipped_metrics": skipped_metrics,
            "passed": bool(scores) and all(v >= 0.5 for v in scores.values()),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        eval_results.append(result)

    # 결과 JSON 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    return eval_results


def batch_evaluate_langfuse(
    *,
    tags: list[str] | None = None,
    from_hours_ago: int = 24,
    limit: int = 200,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
    sample_seed: int = 42,
    skip_scored_prefix: str | None = None,
    metric_categories: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Langfuse External Evaluation Pipeline 전체 구현 — 트레이스 배치 평가.

    Langfuse 공식 문서가 권장하는 External Evaluation Pipeline의 3단계를
    하나의 함수로 오케스트레이션합니다:
    Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    Pipeline 실행 흐름:
        1단계 Fetch  — fetch_traces()로 Langfuse에서 프로덕션 트레이스 조회
                       시간 범위(from_hours_ago)와 태그(tags)로 필터링
                       Ref: https://langfuse.com/docs/query-traces
        2단계 Evaluate — trace_to_testcase()로 LLMTestCase 변환 후
                         MetricsRegistry의 메트릭들로 measure() 실행
                         DeepEval의 GEval/RAG 메트릭이 CoT 기반 LLM 평가 수행
                         Ref: https://deepeval.com/docs/metrics-llm-evals
        3단계 Push   — push_scores()로 "deepeval.*" 접두사 스코어를 Langfuse에 기록
                       Langfuse 대시보드에서 시계열 추적·필터링 가능
                       Ref: https://langfuse.com/docs/scores/custom
        + 보관      — 로컬 JSON 파일에도 결과 저장 (감사 추적용)

    Langfuse 공식 문서의 프로덕션 배포 권장사항:
        - 배치 처리: 10개 단위로 메모리 관리 + 체크포인트 재시작
        - 스케줄링: cron (예: 매일 새벽 5시) 또는 웹훅 트리거
        - 필터링: 태그 기반으로 평가 대상 트레이스 선별
        Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines

    Args:
        tags: Langfuse trace 필터 태그 (예: ["env:prod", "version:0.1.0"])
            enrich_trace()/build_langchain_config()에서 설정한 태그와 매칭
            Ref: https://langfuse.com/docs/tracing-features/tags
        from_hours_ago: 몇 시간 전부터 가져올지 (기본 24)
            cron 주기와 맞추면 평가 누락 방지 (예: 24시간 cron → 24시간 조회)
        limit: 최대 trace 수 (기본 200)
        sample_ratio: 조회된 trace에서 평가할 비율(0.0~1.0).
            None이면 비율 샘플링 미적용.
        max_sample_size: 샘플 최대 건수 상한. None이면 상한 없음.
        sample_seed: deterministic 샘플링 시드 (기본 42)
        skip_scored_prefix: 해당 prefix 스코어가 이미 있는 trace는 제외.
            예: "deepeval" 지정 시 deepeval.*가 있는 trace를 스킵.
        metric_categories: 사용할 메트릭 카테고리 (기본: ["rag", "custom"])
            가능한 값: "rag" (4개), "agent" (2개), "custom" (7개), "all" (전체)
        output_path: 결과 저장 JSON 경로 (기본: data/eval_results/langfuse_batch_results.json)

    Returns:
        평가 결과 리스트. 각 항목은:
            - trace_id: Langfuse trace ID (push_scores에서 사용한 ID)
            - input: 질문 텍스트 (200자 제한, 미리보기용)
            - scores: {메트릭명: 점수} — DeepEval metric.score 값 (0~1)
            - skipped_metrics: 필수 필드 부족으로 스킵된 메트릭 리스트
            - passed: 실행된 메트릭이 있고, 모든 점수가 0.5 이상이면 True
            - timestamp: 평가 시각 (ISO 형식)
    """
    settings = get_settings()
    output_path = output_path or (
        settings.data_dir / "eval_results" / "langfuse_batch_results.json"
    )

    # ── Pipeline 1단계: Fetch ──────────────────────────────────
    # Langfuse에서 프로덕션 트레이스를 시간 범위 + 태그로 필터링하여 조회
    # 내부적으로 lf.fetch_traces(tags, from_timestamp, limit) 호출
    # Ref: https://langfuse.com/docs/query-traces
    traces = fetch_traces(
        tags=tags,
        from_hours_ago=from_hours_ago,
        limit=limit,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
        sample_seed=sample_seed,
        exclude_scored_prefix=skip_scored_prefix,
    )
    if not traces:
        print("[BatchEval] Langfuse에서 트레이스를 찾을 수 없습니다")
        return []

    # ── Pipeline 2단계 준비: 메트릭 인스턴스 생성 ────────────────
    # MetricsRegistry에서 RAG/Agent/Custom 메트릭을 카테고리별로 가져옴
    # 각 메트릭은 DeepEval의 metric.measure(test_case)로 평가 실행
    # Ref: https://deepeval.com/docs/metrics-llm-evals
    registry = get_registry()
    categories = metric_categories or ["rag", "custom"]
    metrics = []
    for cat in categories:
        metrics.extend(registry.get_metrics_by_category(cat))

    results = []
    for trace in traces:
        # ── Pipeline 2단계: Evaluate ───────────────────────────
        # trace_to_testcase(): Langfuse trace → DeepEval LLMTestCase 변환
        # Langfuse 공식 문서 패턴:
        #   LLMTestCase(input=trace.input["args"], actual_output=trace.output)
        # Ref: https://langfuse.com/docs/scores/external-evaluation-pipelines
        test_case = trace_to_testcase(trace)
        if test_case is None:
            continue

        # _run_metrics_on_testcase(): 각 메트릭의 measure() 호출
        # DeepEval의 GEval은 CoT 프롬프팅으로 평가 기준을 생성하고
        # metric.score (0~1)와 metric.reason을 반환
        # Ref: https://deepeval.com/docs/metrics-llm-evals
        scores, skipped_metrics = _run_metrics_on_testcase(test_case, metrics)
        trace_id = trace.id if hasattr(trace, "id") else ""

        # ── Pipeline 3단계: Push ───────────────────────────────
        # DeepEval 평가 결과를 "deepeval.*" 접두사 스코어로 Langfuse에 기록
        # Langfuse 공식 문서 패턴:
        #   langfuse.create_score(trace_id, name, value, comment)
        # → 이 함수는 push_scores()로 자동화하여 여러 메트릭을 한 번에 기록
        # Ref: https://langfuse.com/docs/scores/custom
        push_scores(trace_id, scores)  # type: ignore[arg-type]

        result = {
            "trace_id": trace_id,
            "input": test_case.input[:200],  # 미리보기용 200자 제한
            "scores": scores,
            "skipped_metrics": skipped_metrics,
            "passed": bool(scores) and all(v >= 0.5 for v in scores.values()),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        results.append(result)

    # 결과를 로컬 JSON으로도 저장 (감사 추적 및 오프라인 분석용)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[BatchEval] {len(results)}개 트레이스 평가 완료 → {output_path}")
    return results


def _coerce_numeric_score(value: Any, data_type: str) -> float | None:
    normalized_type = str(data_type or "").upper()

    if normalized_type == "BOOLEAN":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return 1.0
            if lowered in {"false", "0", "no", "n"}:
                return 0.0
        return None

    if normalized_type == "CATEGORICAL":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_prefixed_scores(trace: Any, *, score_prefix: str) -> list[dict[str, Any]]:
    scores = getattr(trace, "scores", None) or []
    prefix = f"{score_prefix}."
    parsed_scores: list[dict[str, Any]] = []

    for score in scores:
        if isinstance(score, dict):
            name = str(score.get("name", "") or "")
            value = score.get("value")
            data_type = str(score.get("data_type", "NUMERIC") or "NUMERIC")
            comment = score.get("comment")
        else:
            name = str(getattr(score, "name", "") or "")
            value = getattr(score, "value", None)
            data_type = str(getattr(score, "data_type", "NUMERIC") or "NUMERIC")
            comment = getattr(score, "comment", None)

        if not name.startswith(prefix):
            continue

        metric_name = name[len(prefix) :]
        parsed_scores.append(
            {
                "full_name": name,
                "metric_name": metric_name,
                "value": value,
                "data_type": data_type.upper(),
                "comment": str(comment) if comment is not None else "",
                "numeric_value": _coerce_numeric_score(value, data_type),
            }
        )

    return parsed_scores


def _resolve_score_threshold(
    *,
    full_name: str,
    metric_name: str,
    default_threshold: float,
    metric_thresholds: dict[str, float] | None,
) -> float:
    if not metric_thresholds:
        return default_threshold
    if full_name in metric_thresholds:
        return float(metric_thresholds[full_name])
    if metric_name in metric_thresholds:
        return float(metric_thresholds[metric_name])
    return default_threshold


def monitor_langfuse_scores(
    *,
    tags: list[str] | None = None,
    from_hours_ago: int = 24,
    limit: int = 200,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
    sample_seed: int = 42,
    score_prefix: str = "eval",
    default_threshold: float = 0.7,
    metric_thresholds: dict[str, float] | None = None,
    require_score_prefix: bool = True,
    output_path: Path | None = None,
    failed_output_path: Path | None = None,
) -> dict[str, Any]:
    """Langfuse 내장 평가 스코어를 샘플링/집계하여 실패 샘플을 추출합니다.

    이 함수는 DeepEval을 호출하지 않습니다.
    Langfuse에서 이미 계산된 score(prefix 기반)를 읽어 pass/fail을 판정합니다.

    Args:
        tags: Langfuse trace 필터 태그
        from_hours_ago: 조회 시간 범위
        limit: 조회 최대 trace 수
        sample_ratio: 조회 결과에서 평가할 비율(0.0~1.0)
        max_sample_size: 샘플 최대 건수
        sample_seed: deterministic 샘플링 시드
        score_prefix: 평가 스코어 접두사(예: "eval", "quality")
        default_threshold: 기본 실패 임계값
        metric_thresholds: 메트릭별 임계값 override
            키는 full_name(`eval.faithfulness`) 또는 suffix(`faithfulness`) 둘 다 허용
        require_score_prefix: True면 prefix score가 있는 trace만 결과에 포함
        output_path: 모니터링 스냅샷 JSON 저장 경로
        failed_output_path: 실패 샘플 전용 JSON 저장 경로

    Returns:
        {"summary": {...}, "samples": [...], "failed_samples": [...]}
    """
    settings = get_settings()
    output_path = output_path or (
        settings.data_dir / "eval_results" / "langfuse_monitoring_snapshot.json"
    )
    failed_output_path = failed_output_path or (
        settings.data_dir / "eval_results" / "langfuse_failed_samples.json"
    )

    traces = fetch_traces(
        tags=tags,
        from_hours_ago=from_hours_ago,
        limit=limit,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
        sample_seed=sample_seed,
    )
    if not traces:
        summary = {
            "score_prefix": score_prefix,
            "default_threshold": default_threshold,
            "total_sampled": 0,
            "total_collected": 0,
            "traces_with_scores": 0,
            "traces_without_scores": 0,
            "traces_evaluated": 0,
            "evaluated_scores": 0,
            "traces_failed": 0,
            "failure_rate": 0.0,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        snapshot: dict[str, Any] = {
            "summary": summary,
            "samples": [],
            "failed_samples": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(snapshot, file, ensure_ascii=False, indent=2)
        with open(failed_output_path, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)
        print("[LangfuseMonitor] Langfuse에서 트레이스를 찾을 수 없습니다")
        return snapshot

    samples: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []

    traces_with_scores = 0
    traces_without_scores = 0
    traces_evaluated = 0
    evaluated_scores = 0

    for trace in traces:
        prefixed_scores = _extract_prefixed_scores(trace, score_prefix=score_prefix)
        has_scores = bool(prefixed_scores)
        if has_scores:
            traces_with_scores += 1
        else:
            traces_without_scores += 1
            if require_score_prefix:
                continue

        test_case = trace_to_testcase(trace)
        input_text = test_case.input if test_case else ""
        output_text = test_case.actual_output if test_case else ""
        context = test_case.context if test_case else None
        retrieval_context = test_case.retrieval_context if test_case else None

        scores: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        failed_metrics: list[str] = []
        unevaluable_metrics: list[str] = []
        score_comments: dict[str, str] = {}
        evaluated_metric_count_for_trace = 0

        for item in prefixed_scores:
            metric_name = item["metric_name"]
            full_name = item["full_name"]
            score_value = item["value"]
            numeric_value = item["numeric_value"]
            scores[metric_name] = score_value
            if item["comment"]:
                score_comments[metric_name] = item["comment"]

            threshold = _resolve_score_threshold(
                full_name=full_name,
                metric_name=metric_name,
                default_threshold=default_threshold,
                metric_thresholds=metric_thresholds,
            )
            thresholds[metric_name] = threshold

            if numeric_value is None:
                unevaluable_metrics.append(metric_name)
                continue

            evaluated_metric_count_for_trace += 1
            evaluated_scores += 1
            if numeric_value < threshold:
                failed_metrics.append(metric_name)

        if evaluated_metric_count_for_trace > 0:
            traces_evaluated += 1

        trace_id = str(getattr(trace, "id", "") or "")
        sample = {
            "trace_id": trace_id,
            "input": input_text,
            "actual_output": output_text,
            "context": context,
            "retrieval_context": retrieval_context,
            "scores": scores,
            "thresholds": thresholds,
            "failed_metrics": sorted(set(failed_metrics)),
            "unevaluable_metrics": sorted(set(unevaluable_metrics)),
            "score_comments": score_comments,
            "passed": evaluated_metric_count_for_trace > 0 and not failed_metrics,
            "evaluated_metric_count": evaluated_metric_count_for_trace,
            "has_scores": has_scores,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        samples.append(sample)
        if failed_metrics:
            failed_samples.append(sample)

    failure_rate = (len(failed_samples) / len(samples)) if samples else 0.0
    summary = {
        "score_prefix": score_prefix,
        "default_threshold": default_threshold,
        "metric_threshold_overrides": metric_thresholds or {},
        "total_sampled": len(traces),
        "total_collected": len(samples),
        "traces_with_scores": traces_with_scores,
        "traces_without_scores": traces_without_scores,
        "traces_evaluated": traces_evaluated,
        "evaluated_scores": evaluated_scores,
        "traces_failed": len(failed_samples),
        "failure_rate": failure_rate,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    snapshot = {
        "summary": summary,
        "samples": samples,
        "failed_samples": failed_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)

    with open(failed_output_path, "w", encoding="utf-8") as file:
        json.dump(failed_samples, file, ensure_ascii=False, indent=2)

    print(
        "[LangfuseMonitor] "
        f"sampled={len(traces)}, collected={len(samples)}, failed={len(failed_samples)} "
        f"→ {output_path}"
    )
    return snapshot
