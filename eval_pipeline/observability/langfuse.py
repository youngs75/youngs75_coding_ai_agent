"""Langfuse 관측성 유틸리티 모듈.

Langfuse 클라이언트 초기화, 활성화 여부 확인, 그리고
트레이스 강화(Enrichment) 유틸리티를 제공합니다.

배경:
    LangChain 콜백 핸들러는 기본적으로 입력/출력, 토큰 사용량, 레이턴시 등을
    자동 수집합니다. 하지만 프로덕션 환경에서는 "누가(user_id)", "어떤 세션에서
    (session_id)", "어떤 환경인지(tags/metadata)" 등의 부가 컨텍스트와
    다양한 타입의 커스텀 스코어가 필요합니다.

    이 모듈은 Langfuse 공식 문서가 제안하는 두 가지 트레이스 강화 방식을
    모두 지원하는 유틸리티를 제공합니다.

기본 기능 (기존):
    from youngs75_a2a.eval_pipeline.observability.langfuse import enabled, client

    if enabled():
        lf = client()
        lf.score(trace_id="...", name="metric", value=0.8)

트레이스 강화 (신규) — 두 가지 방식 지원:

    방법 1 - propagate_attributes 래핑 (권장, Langfuse v3):
        Langfuse SDK v3의 네이티브 방식. OpenTelemetry 컨텍스트를 통해
        모든 하위 span에 속성을 자동 전파합니다.
        Ref: https://langfuse.com/docs/sdk/python/decorators

        from youngs75_a2a.eval_pipeline.observability.langfuse import enrich_trace

        with enrich_trace(user_id="user-123", session_id="sess-abc",
                          tags=["prod", "rag-v2"]):
            result = agent.invoke(input_state, config=cfg)

    방법 2 - LangChain config metadata:
        LangChain invoke() 호출 시 config.metadata에 langfuse_* 키를 전달하여
        콜백 핸들러가 Langfuse 트레이스 속성으로 변환합니다.
        Ref: https://langfuse.com/docs/integrations/langchain/tracing

        from youngs75_a2a.eval_pipeline.observability.langfuse import build_langchain_config

        config = build_langchain_config(user_id="user-123", callbacks=[handler])
        result = agent.invoke(input_state, config=config)

    스코어링 (다중 데이터 타입):
        Langfuse는 NUMERIC, BOOLEAN, CATEGORICAL 세 가지 스코어 타입을 지원합니다.
        Ref: https://langfuse.com/docs/scores/custom

        from youngs75_a2a.eval_pipeline.observability.langfuse import score_trace

        score_trace(trace_id, name="metric", value=0.85)
        score_trace(trace_id, name="passed", value=True, data_type="BOOLEAN")
        score_trace(trace_id, name="level", value="high", data_type="CATEGORICAL")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Literal

from langfuse import get_client, propagate_attributes

from youngs75_a2a.eval_pipeline.settings import Settings, get_settings


def enabled(settings: Settings | None = None) -> bool:
    """Langfuse가 올바르게 설정되어 사용 가능한지 확인합니다.

    langfuse_tracing_enabled 플래그가 True이고,
    host, public_key, secret_key가 모두 비어있지 않아야 True를 반환합니다.
    이 함수를 사용하여 Langfuse 관련 코드를 조건부로 실행하세요.

    Args:
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    Returns:
        bool: Langfuse 사용 가능 여부
    """
    s = settings or get_settings()
    return bool(
        s.langfuse_tracing_enabled
        and s.langfuse_host
        and s.langfuse_public_key
        and s.langfuse_secret_key
    )


def client():
    """Langfuse 클라이언트 싱글턴을 반환합니다.

    langfuse 라이브러리의 get_client()를 래핑합니다.
    환경변수에서 자동으로 연결 정보를 읽어 초기화합니다.

    Returns:
        Langfuse: Langfuse 클라이언트 인스턴스

    Warning:
        enabled()가 False인 상태에서 호출하면 예외가 발생할 수 있습니다.
        반드시 enabled() 확인 후 사용하세요.
    """
    return get_client()


# ── 트레이스 강화(Enrichment) 유틸리티 ──────────────────────


def default_metadata(settings: Settings | None = None) -> dict[str, str]:
    """Settings에서 기본 메타데이터를 구성합니다.

    Langfuse의 metadata는 트레이스/observation에 부착되는 key-value 쌍으로,
    대시보드에서 필터링·검색·상관분석에 활용됩니다.
    Ref: https://langfuse.com/docs/tracing-features/metadata

    env, service_name, app_version을 Langfuse 메타데이터용 딕셔너리로 반환합니다.
    propagate_attributes()의 metadata 인자에 바로 전달할 수 있습니다.

    Langfuse 제약사항 (공식 문서 기준):
        - 키: 영숫자(alphanumeric)만 허용
        - 값: US-ASCII 문자열, 200자 이내
        - 200자 초과 시 Langfuse SDK가 자동으로 해당 값을 드롭(drop)

    Args:
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    Returns:
        dict[str, str]: Langfuse 메타데이터 딕셔너리

    사용 예시:
        meta = default_metadata()
        # → {"env": "local", "service_name": "agent-ops-day3", "app_version": "0.1.0"}

        # propagate_attributes와 함께 사용:
        with propagate_attributes(metadata=meta):
            ...
    """
    s = settings or get_settings()
    return {
        "env": s.env,
        "service_name": s.service_name,
        "app_version": s.app_version,
    }


def default_tags(settings: Settings | None = None) -> list[str]:
    """Settings에서 기본 태그 목록을 구성합니다.

    Langfuse의 tags는 트레이스/observation을 분류하는 문자열 리스트입니다.
    개별 observation의 태그는 상위 trace로 자동 집계(aggregate)되어,
    대시보드에서 태그 기반 필터링이 가능합니다.
    Ref: https://langfuse.com/docs/tracing-features/tags

    env, service_name, app_version을 "key:value" 형식의 태그 리스트로 반환합니다.
    propagate_attributes()의 tags 인자에 바로 전달하거나,
    LangChain config의 langfuse_tags에 전달할 수 있습니다.

    Langfuse 제약사항:
        - 각 태그는 최대 200자
        - 200자 초과 시 해당 태그가 드롭됨

    Args:
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    Returns:
        list[str]: Langfuse 태그 리스트

    사용 예시:
        tags = default_tags()
        # → ["env:local", "service:agent-ops-day3", "version:0.1.0"]

        # fetch_traces에서 태그 기반 필터링:
        traces = lf.fetch_traces(tags=["env:prod"])
    """
    s = settings or get_settings()
    tags: list[str] = []
    if s.env:
        tags.append(f"env:{s.env}")
    if s.service_name:
        tags.append(f"service:{s.service_name}")
    if s.app_version:
        tags.append(f"version:{s.app_version}")
    return tags


def score_trace(
    trace_id: str,
    *,
    name: str,
    value: float | str | bool,
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "NUMERIC",
    comment: str | None = None,
    observation_id: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Langfuse trace에 스코어를 기록합니다 (모든 데이터 타입 지원).

    Langfuse의 Score는 trace 또는 개별 observation에 부착하는 평가 지표입니다.
    모델 기반 평가(LLM-as-a-Judge), 사용자 피드백, 커스텀 평가 등
    다양한 평가 워크플로우에서 활용됩니다.
    Ref: https://langfuse.com/docs/scores/custom

    지원하는 세 가지 데이터 타입 (Langfuse 공식 스펙):
        - NUMERIC: 연속 수치 (float). 시계열 추적, 평균/분포 분석에 적합.
          예: faithfulness=0.85, latency_ms=1200.0
        - BOOLEAN: 이진 판정 (bool → 내부적으로 1.0/0.0 변환). 통과/실패 필터링에 최적.
          예: schema_valid=True, safety_passed=False
        - CATEGORICAL: 범주형 문자열. 대시보드 그룹핑/필터링에 활용.
          예: risk_level="low", error_code="TIMEOUT"

    Langfuse가 비활성화된 경우 아무 작업도 수행하지 않습니다.

    Args:
        trace_id: Langfuse trace ID
        name: 스코어 이름. 접두사 규칙 권장 (예: "deepeval.faithfulness")
        value: 스코어 값 (위의 데이터 타입별 설명 참조)
        data_type: 스코어 데이터 타입 (기본: "NUMERIC")
        comment: 선택적 코멘트 (대시보드에서 스코어 옆에 표시됨)
        observation_id: 특정 observation(span)에 스코어를 연결할 ID.
            None이면 trace 전체에 스코어가 부착됨.
            Ref: https://langfuse.com/docs/scores/custom
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    사용 예시:
        # NUMERIC 스코어 — 연속 수치로 품질 추적
        score_trace(tid, name="deepeval.faithfulness", value=0.85)

        # BOOLEAN 스코어 — 통과/실패 판정
        score_trace(tid, name="quality.passed", value=True, data_type="BOOLEAN")

        # CATEGORICAL 스코어 — 범주형 분류
        score_trace(tid, name="risk.level", value="low", data_type="CATEGORICAL",
                    comment="위험도 낮음으로 판정")
    """
    s = settings or get_settings()
    if not enabled(s):
        return

    lf = client()

    # 타입별 값 변환 — Langfuse API가 기대하는 형식으로 정규화
    # BOOLEAN은 내부적으로 1.0/0.0 float로 전송되지만,
    # 대시보드에서는 True/False로 표시됨 (Langfuse가 data_type으로 구분)
    score_value: Any
    if data_type == "BOOLEAN":
        score_value = 1.0 if bool(value) else 0.0
    elif data_type == "CATEGORICAL":
        score_value = str(value)
    else:  # NUMERIC
        try:
            score_value = float(value)
        except (TypeError, ValueError):
            score_value = 0.0

    # lf.score()는 Langfuse Python SDK의 스코어 기록 메서드
    # trace_id만 지정하면 trace 레벨, observation_id도 지정하면 span 레벨에 부착
    kwargs: dict[str, Any] = {
        "trace_id": trace_id,
        "name": name,
        "value": score_value,
        "data_type": data_type,
    }
    if comment:
        kwargs["comment"] = comment
    if observation_id:
        kwargs["observation_id"] = observation_id

    lf.create_score(**kwargs)


@contextmanager
def enrich_trace(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    version: str | None = None,
    settings: Settings | None = None,
):
    """Langfuse 트레이스를 풍부한 컨텍스트로 감싸는 컨텍스트 매니저.

    Langfuse SDK v3의 propagate_attributes()를 래핑하여, Settings의
    기본 메타데이터/태그를 자동으로 포함합니다.
    추가 tags/metadata를 전달하면 기본값에 병합됩니다.
    Ref: https://langfuse.com/docs/sdk/python/decorators

    내부적으로 OpenTelemetry 컨텍스트 전파를 사용하므로,
    이 블록 안에서 생성되는 모든 하위 span(@observe, LangChain 콜백 등)에
    user_id, session_id, tags, metadata가 자동 상속됩니다.

    Langfuse가 비활성화된 경우에도 안전하게 동작합니다 (패스스루).

    주의 (Langfuse 공식 문서):
        "Call this as early as possible within your trace/workflow."
        이 컨텍스트 진입 이전에 생성된 span에는 속성이 전파되지 않습니다.
        따라서 에이전트/체인 실행 전에 반드시 먼저 진입해야 합니다.

    propagate_attributes가 전파하는 속성들:
        - user_id: 사용자별 비용/성능 분석 가능 (US-ASCII, 200자 이내)
        - session_id: 멀티턴 대화를 하나의 세션으로 묶음
          Ref: https://langfuse.com/docs/tracing-features/sessions
        - tags: 대시보드에서 필터링/분류용 태그 리스트
        - metadata: key-value 쌍으로 커스텀 차원 추가
        - version: 앱/에이전트 버전 추적
        - trace_name: 대시보드에서 표시될 트레이스 이름

    Args:
        user_id: 사용자 식별자 (예: "user-123")
        session_id: 세션 식별자 (예: "conversation-456")
        trace_name: 트레이스 이름 (예: "agent-ops-day3:remediation")
        tags: 추가 태그 목록 (기본 태그에 병합됨)
        metadata: 추가 메타데이터 (기본 메타데이터에 병합됨)
        version: 앱 버전 (기본: settings.app_version)
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    사용 예시:
        # 기본 사용: user_id + session_id로 사용자 추적
        with enrich_trace(user_id="user-123", session_id="sess-abc",
                          tags=["prod", "rag-v2"],
                          metadata={"experiment": "prompt_v3"}):
            result = agent.invoke(input_state, config=cfg)

        # Langfuse가 비활성화되어도 안전하게 동작 (패스스루):
        with enrich_trace(user_id="user-123"):
            result = agent.invoke(input_state)  # Langfuse 미설정 시 그냥 통과

    Yields:
        None (컨텍스트 매니저)
    """
    s = settings or get_settings()

    # Langfuse 비활성화 시 패스스루
    if not enabled(s):
        yield
        return

    # 기본값(Settings 기반)과 사용자 지정값을 병합
    # metadata: default_metadata()가 env/service_name/app_version을 제공하고,
    #           사용자가 전달한 metadata가 이를 덮어쓰거나 확장
    merged_metadata = default_metadata(s)
    if metadata:
        merged_metadata.update(metadata)

    # tags: default_tags()가 env:xxx/service:xxx/version:xxx를 제공하고,
    #        사용자가 전달한 tags가 리스트에 추가됨
    merged_tags = default_tags(s)
    if tags:
        merged_tags.extend(tags)

    # propagate_attributes(): Langfuse SDK v3의 핵심 메커니즘
    # OpenTelemetry 컨텍스트에 속성을 설정하여 모든 하위 span에 자동 전파
    # Ref: https://langfuse.com/docs/sdk/python/decorators
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        trace_name=trace_name or f"{s.service_name}:{session_id or 'default'}",
        metadata=merged_metadata,
        tags=merged_tags,
        version=version or s.app_version,
    ):
        yield


def build_langchain_config(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, str] | None = None,
    callbacks: list | None = None,
    settings: Settings | None = None,
) -> dict:
    """LangChain invoke용 config 딕셔너리를 구성합니다.

    enrich_trace()의 대안으로, LangChain의 invoke/batch/stream 호출 시
    config 파라미터를 통해 Langfuse 트레이스 속성을 전달하는 방식입니다.
    Ref: https://langfuse.com/docs/integrations/langchain/tracing

    Langfuse의 LangChain CallbackHandler는 config.metadata 내의 특수 키를
    인식하여 자동으로 트레이스 속성으로 변환합니다:
        - langfuse_user_id → Langfuse trace의 user_id
        - langfuse_session_id → Langfuse trace의 session_id
        - langfuse_tags → Langfuse trace의 tags (List[str])
        - langfuse_trace_id → 기존 trace에 연결 (동일 trace에 여러 호출 연결)

    enrich_trace() vs build_langchain_config() 선택 기준:
        - enrich_trace(): LangChain 외의 코드도 포함하는 넓은 컨텍스트에 적합.
          propagate_attributes 기반이라 모든 하위 span에 자동 전파.
        - build_langchain_config(): 특정 chain.invoke() 호출에만 속성을
          전달하고 싶을 때 적합. 기존 LangChain 코드에 최소한의 변경.

    Args:
        user_id: Langfuse 사용자 ID
        session_id: Langfuse 세션 ID
        trace_id: 기존 trace에 연결할 ID
        tags: 추가 태그 (기본 태그에 병합됨)
        extra_metadata: 추가 메타데이터 (config.metadata에 병합됨)
        callbacks: LangChain 콜백 핸들러 리스트 (예: [langfuse_handler])
        settings: Settings 인스턴스. None이면 get_settings()로 자동 로드

    Returns:
        dict: chain.invoke()의 config 인자로 전달할 딕셔너리

    사용 예시:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        config = build_langchain_config(
            user_id="user-123",
            session_id="sess-abc",
            callbacks=[handler],
            tags=["prod"],
        )
        # chain.invoke()에 config 전달 → Langfuse가 user_id, session_id 등 인식
        result = agent.invoke(input_state, config=config)
    """
    s = settings or get_settings()

    # Settings 기반 기본 태그 + 사용자 추가 태그 병합
    merged_tags = default_tags(s)
    if tags:
        merged_tags.extend(tags)

    # LangChain invoke config의 metadata에 langfuse_* 키를 설정
    # Langfuse CallbackHandler가 이 키들을 인식하여 트레이스 속성으로 변환
    # Ref: https://langfuse.com/docs/integrations/langchain/tracing
    lf_metadata: dict[str, Any] = {}
    if extra_metadata:
        lf_metadata.update(extra_metadata)

    # langfuse_user_id: 사용자별 비용·성능 분석용
    if user_id:
        lf_metadata["langfuse_user_id"] = user_id
    # langfuse_session_id: 멀티턴 대화를 하나의 세션으로 그룹핑
    if session_id:
        lf_metadata["langfuse_session_id"] = session_id
    # langfuse_trace_id: 기존 trace에 연결하여 여러 호출을 하나의 trace로 묶음
    if trace_id:
        lf_metadata["langfuse_trace_id"] = trace_id
    # langfuse_tags: 대시보드 필터링·분류용 태그 리스트
    if merged_tags:
        lf_metadata["langfuse_tags"] = merged_tags

    config: dict[str, Any] = {"metadata": lf_metadata}
    if callbacks:
        config["callbacks"] = callbacks

    return config
