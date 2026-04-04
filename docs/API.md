# API 레퍼런스

**패키지**: `youngs75_a2a`  
**Python**: 3.13  
**import 경로**: `from youngs75_a2a.{모듈} import {클래스}`

---

## 1. Core 모듈 (`youngs75_a2a.core`)

### 1.1 BaseGraphAgent

**경로**: `youngs75_a2a.core.base_agent`

LangGraph 그래프 에이전트의 기본 클래스. 모든 에이전트는 이 클래스를 상속한다.

```python
class BaseGraphAgent:
    NODE_NAMES: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *,
        config: BaseAgentConfig | None = None,
        model: BaseChatModel | None = None,
        state_schema: type | None = None,
        config_schema: type | None = None,
        input_state: type | None = None,
        output_state: type | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        max_retry_attempts: int = 2,
        agent_name: str | None = None,
        debug: bool = False,
        auto_build: bool = True,
    ) -> None: ...

    @classmethod
    async def create(cls, **kwargs) -> "BaseGraphAgent": ...

    async def async_init(self) -> None: ...
    def build_graph(self) -> None: ...
    def init_nodes(self, graph: StateGraph) -> None: ...   # 추상
    def init_edges(self, graph: StateGraph) -> None: ...   # 추상
    def get_node_name(self, key: str) -> str: ...
```

| 메서드 | 설명 |
|--------|------|
| `create(**kwargs)` | 비동기 팩토리. `async_init()` 호출 후 `build_graph()` 실행 |
| `async_init()` | 비동기 초기화 훅 (MCP 도구 로딩 등). 서브클래스에서 오버라이드 |
| `build_graph()` | `StateGraph`를 빌드하고 컴파일. `init_nodes()` + `init_edges()` 호출 |
| `init_nodes(graph)` | 그래프에 노드 등록. **서브클래스에서 반드시 구현** |
| `init_edges(graph)` | 그래프에 엣지 등록. **서브클래스���서 반드시 구현** |
| `get_node_name(key)` | `NODE_NAMES` 딕셔너리에서 노드 이름 조��� |

### 1.2 BaseAgentConfig

**경로**: `youngs75_a2a.core.config`

모든 에이전트 Config의 기본 클래스. Pydantic BaseModel 기반.

```python
class BaseAgentConfig(BaseModel):
    model_provider: str          # 기본: "openai" (환경변수 MODEL_PROVIDER)
    default_model: str           # 기본: "gpt-5.4" (환경변수 MODEL_NAME)
    temperature: float           # 기본: 0.1 (환경변수 TEMPERATURE)
    max_retries: int             # 기본: 3
    mcp_servers: dict[str, str]  # MCP 서버 엔드포인트 매핑
    skills_dir: str | None       # 스킬 파일 디렉토리 (환경변수 SKILLS_DIR)
    model_tiers: dict[str, TierConfig]      # 멀티티어 모델 설정
    purpose_tiers: dict[str, str]           # purpose -> tier 매핑

    def get_model(self, purpose: str = "default", *, structured: type | None = None) -> BaseChatModel: ...
    def get_tier_config(self, purpose: str = "default") -> TierConfig: ...
    def get_mcp_endpoint(self, server_name: str) -> str | None: ...
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "BaseAgentConfig": ...
    def to_langgraph_configurable(self) -> dict[str, Any]: ...
```

| 메서드 | 설명 |
|--------|------|
| `get_model(purpose)` | 목적별 LLM 모델 반환. 레거시 경로 사용 (서브클래스에서 티어 경로 오버라이드 가���) |
| `get_tier_config(purpose)` | 목적에 해당하는 `TierConfig` 반환 |
| `get_mcp_endpoint(server_name)` | MCP 서버 URL 조회 |
| `from_runnable_config(config)` | LangGraph `RunnableConfig`에서 설정 추출 |
| `to_langgraph_configurable()` | LangGraph configurable dict로 변환 |

### 1.3 ModelTiers

**경로**: `youngs75_a2a.core.model_tiers`

멀티티어 모델 해석 시스템.

```python
class ModelTier(str, Enum):
    STRONG = "strong"
    DEFAULT = "default"
    FAST = "fast"

class TierConfig(BaseModel):
    model: str                     # 모델명
    provider: str = "openai"       # 프로바이더
    context_window: int = 128_000  # 컨텍스트 윈도우 크기
    temperature: float | None = None

    @property
    def summarization_threshold(self) -> int: ...  # context_window * 0.75

def build_default_tiers() -> dict[str, TierConfig]: ...
def build_default_purpose_tiers() -> dict[str, str]: ...
def resolve_tier_config(purpose, tiers, purpose_tiers) -> TierConfig: ...
def create_chat_model(tier_config, *, temperature=0.1, structured=None, **kwargs) -> BaseChatModel: ...
```

### 1.4 MCPToolLoader

**경로**: `youngs75_a2a.core.mcp_loader`

MCP 서버에서 도구를 로딩하고 캐싱하는 클���스.

```python
class MCPToolLoader:
    def __init__(
        self,
        servers: dict[str, str],           # {"서버명": "http://host:port/mcp/"}
        transport: str = "streamable_http",
        health_timeout: float = 3.0,
        max_retries: int = 2,
    ): ...

    @property
    def is_loaded(self) -> bool: ...
    async def load(self, *, force: bool = False) -> list[Any]: ...
    def get_tool_descriptions(self) -> str: ...
    def reset(self) -> None: ...
```

| 메서드 | 설명 |
|--------|------|
| `load(force=False)` | MCP 도구 로딩. 캐시가 있으면 반환, 없으면 헬스체크 후 로딩 |
| `get_tool_descriptions()` | 로딩된 도구의 설명 문자열 반환 |
| `reset()` | 캐시 초기화 |

**Graceful Degradation**: 모든 MCP 서버가 불능이면 빈 도구 목록으로 진행.

### 1.5 BaseGraphState

**경로**: `youngs75_a2a.core.base_state`

```python
class BaseGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 1.6 ActionValidator

**경로**: `youngs75_a2a.core.action_validator`

Safety Envelope. 에이전트 출력을 실행 전에 검증한다.

```python
class ActionValidator:
    def __init__(
        self,
        *,
        allowed_extensions: list[str] | None = None,   # 기본: .py, .js, .ts 등 8종
        max_delete_lines: int = 100,
        allowed_directories: list[str] | None = None,
    ): ...

    def validate(self, code: str, **context) -> ValidationReport: ...

class ValidationReport:
    results: list[ValidationResult]
    @property
    def is_safe(self) -> bool: ...
    @property
    def has_warnings(self) -> bool: ...
    @property
    def blocked_rules(self) -> list[str]: ...
    def summary(self) -> str: ...

class ValidationStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
```

**검증 규칙**:

| 규칙 | 상태 | 설명 |
|------|------|------|
| `secret_exposure` | BLOCK | API 키, 패스워드 등 시크릿 하드코딩 감지 |
| `dangerous_command` | BLOCK | `rm -rf /`, `DROP TABLE` 등 위험 명령 |
| `delete_volume` | WARN | 대량 삭제 지표 초과 |
| `file_extension` | BLOCK | 허용 확장자 외 파일 수정 |
| `directory_scope` | BLOCK | 허용 디렉토리 범위 밖 파일 접근 |

### 1.7 override_reducer

**경로**: `youngs75_a2a.core.reducers`

```python
def override_reducer(current_value: Any, new_value: Any) -> Any:
    """
    new_value가 {"type": "override", "value": ...} 형태이면 덮어쓰기.
    그 외에는 리스트 병합.
    """
```

### 1.8 tool_call_utils

**경로**: `youngs75_a2a.core.tool_call_utils`

LangChain, OpenAI, dict 등 다양한 형태의 tool_call 객체를 안전하게 처리.

```python
def tc_name(tool_call: Any) -> str | None: ...   # 도구 이름 추출
def tc_id(tool_call: Any) -> str | None: ...     # 도구 호출 ID 추출
def tc_args(tool_call: Any) -> dict[str, Any]: ... # 도구 인자 추출 (JSON 파싱 포함)
```

### 1.9 Memory 시스템

**경로**: `youngs75_a2a.core.memory`

#### MemoryItem / MemoryType

```python
class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryItem(BaseModel):
    id: str                      # UUID hex
    type: MemoryType
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    session_id: str | None
    score: float

    def matches_tags(self, query_tags: list[str]) -> bool: ...
```

#### MemoryStore

```python
class MemoryStore:
    def __init__(self, store=None, search=None): ...

    def put(self, item: MemoryItem) -> None: ...
    def get(self, item_id, memory_type, session_id=None) -> MemoryItem | None: ...
    def search(self, query, *, memory_type=None, tags=None, session_id=None, limit=10) -> list[MemoryItem]: ...
    def list_by_type(self, memory_type, session_id=None) -> list[MemoryItem]: ...
    def delete(self, item_id, memory_type, session_id=None) -> bool: ...
    def clear(self, memory_type=None) -> int: ...
    def accumulate_skill(self, code, description, tags=None, *, novelty_threshold=0.7) -> MemoryItem | None: ...
    def retrieve_skills(self, query, *, tags=None, limit=5) -> list[MemoryItem]: ...

    @property
    def total_count(self) -> int: ...
```

#### MemoryAwareState

```python
class MemoryAwareState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    semantic_context: Annotated[list[str], override_reducer]
    episodic_log: Annotated[list[str], override_reducer]
    procedural_context: Annotated[list[str], override_reducer]
```

#### TwoStageSearch

```python
class TwoStageSearch:
    """GAM 이중 구조: 태그 기반 필터링 -> BM25 컨텐츠 순위."""
    def __init__(self, tag_limit=50, final_limit=10): ...
    def search(self, query, candidates) -> list[MemoryItem]: ...
```

### 1.10 Skills 시스템

**경로**: `youngs75_a2a.core.skills`

#### Skill / SkillMetadata / SkillLevel

```python
class SkillLevel(IntEnum):
    L1_METADATA = 1
    L2_BODY = 2
    L3_REFERENCES = 3

class SkillMetadata(BaseModel):
    name: str
    description: str
    tags: list[str]
    version: str = "1.0.0"
    enabled: bool = True

class Skill(BaseModel):
    metadata: SkillMetadata
    body: str | None                    # L2
    references: list[SkillReference]    # L3
    source_path: Path | None

    @property
    def name(self) -> str: ...
    @property
    def loaded_level(self) -> SkillLevel: ...
    def as_context_entry(self) -> str: ...
```

#### SkillLoader

```python
class SkillLoader:
    def __init__(self, skills_dir: Path | str): ...
    def discover(self) -> list[Skill]: ...                              # L1 탐색
    def load(self, name, level=SkillLevel.L2_BODY) -> Skill | None: ...  # 이름으로 로드
    def load_references(self, skill) -> Skill: ...                       # L3 참조 로드
```

#### SkillRegistry

```python
class SkillRegistry:
    def __init__(self, loader=None): ...
    def discover(self) -> list[str]: ...
    def register(self, skill) -> None: ...
    def get(self, name) -> Skill | None: ...
    def activate(self, name, *, with_refs=False) -> Skill | None: ...
    def get_context_entries(self) -> list[str]: ...
    def list_skills(self) -> list[Skill]: ...
    def search_by_tags(self, tags) -> list[Skill]: ...
    @property
    def activation_stats(self) -> dict[str, int]: ...
```

### 1.11 SubAgents 시스템

**경로**: `youngs75_a2a.core.subagents`

```python
class SubAgentSpec(BaseModel):
    name: str
    description: str
    capabilities: list[str]
    endpoint: str | None         # A2A 엔드포인트 (None이면 로컬)
    model_tier: str = "default"
    cost_weight: float = 1.0
    status: SubAgentStatus       # AVAILABLE | BUSY | FAILED | DISABLED

class SubAgentRegistry:
    def __init__(self, cost_sensitivity: float = 0.3): ...
    def register(self, spec) -> None: ...
    def select(self, task_type, required_capabilities=None) -> SelectionResult | None: ...
    def record_usage(self, record) -> None: ...
    def get_success_rate(self, agent_name, task_type=None) -> float: ...
    @property
    def usage_stats(self) -> dict[str, dict[str, int]]: ...
```

---

## 2. Agents 모듈 (`youngs75_a2a.agents`)

### 2.1 CodingAssistantAgent

**경로**: `youngs75_a2a.agents.coding_assistant.agent`

```python
class CodingAssistantAgent(BaseGraphAgent):
    NODE_NAMES = {
        "PARSE": "parse_request",
        "EXECUTE": "execute_code",
        "EXECUTE_TOOLS": "execute_tools",
        "VERIFY": "verify_result",
    }

    def __init__(
        self,
        *,
        config: CodingConfig | None = None,
        model: BaseChatModel | None = None,
        memory_store: MemoryStore | None = None,
        **kwargs,
    ) -> None: ...
```

**CodingConfig** (`youngs75_a2a.agents.coding_assistant.config`):

```python
class CodingConfig(BaseAgentConfig):
    generation_model: str | None     # 환경변수 CODING_GEN_MODEL
    verification_model: str | None   # 환경변수 CODING_VERIFY_MODEL
    allowed_extensions: list[str]    # 허용 파일 확장자
    max_delete_lines: int = 100
    mcp_servers: dict[str, str]      # {"code_tools": "http://localhost:3003/mcp/"}
    max_tool_calls: int = 10
    purpose_tiers: dict[str, str]    # generation->STRONG, verification->DEFAULT, parsing->FAST
```

**CodingState** (`youngs75_a2a.agents.coding_assistant.schemas`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `messages` | `list[BaseMessage]` | 대화 이력 |
| `semantic_context` | `list[str]` | Semantic Memory |
| `skill_context` | `list[str]` | Skills L1 메타데이터 |
| `episodic_log` | `list[str]` | Episodic Memory |
| `procedural_skills` | `list[str]` | Procedural Memory |
| `parse_result` | `ParseResult` | 요청 분석 결과 |
| `generated_code` | `str` | 생성된 코드 |
| `verify_result` | `VerifyResult` | 검증 결과 |
| `project_context` | `list[str]` | JIT 원본 참조 |
| `iteration` | `int` | 현재 반복 횟수 |
| `max_iterations` | `int` | 최대 반복 횟수 (기본: 3) |

**PromptRegistry** (`youngs75_a2a.agents.coding_assistant.prompts`):

```python
class PromptRegistry:
    def get_prompt(self, name, *, version=None) -> str: ...
    def get_current_version(self, name) -> str: ...
    def list_versions(self, name) -> list[str]: ...
    def list_prompts(self) -> list[str]: ...
    def apply_remediation(self, changes) -> list[str]: ...

def get_prompt_registry() -> PromptRegistry: ...    # 싱글턴
def reset_prompt_registry() -> None: ...            # 초기화 (테스트용)
```

### 2.2 DeepResearchAgent

**경로**: `youngs75_a2a.agents.deep_research.agent`

```python
class DeepResearchAgent(BaseGraphAgent):
    NODE_NAMES = {
        "CLARIFY": "clarify_with_user",
        "BRIEF": "write_research_brief",
        "SUPERVISOR": "research_supervisor",
        "REPORT": "final_report_generation",
    }
```

**ResearchConfig** (`youngs75_a2a.agents.deep_research.config`):

| 필드 | 환경변수 | 기본값 | 설명 |
|------|----------|--------|------|
| `allow_clarification` | `ALLOW_CLARIFICATION` | `true` | 질문 명확화 허용 |
| `max_concurrent_research_units` | `MAX_CONCURRENT_RESEARCH` | `3` | 병렬 연구 동시성 제한 |
| `max_researcher_iterations` | `MAX_RESEARCHER_ITERATIONS` | `3` | 연구자 최대 반복 |
| `enable_hitl` | `ENABLE_HITL` | `false` | HITL 승인 루프 활성화 |
| `research_model` | `MODEL_NAME` | `gpt-5.4` | 연구 수행 모델 |
| `compression_model` | `COMPRESSION_MODEL` | `gpt-4o-2024-11-20` | 결과 압축 모델 |
| `final_report_model` | `FINAL_REPORT_MODEL` | `gpt-5.4` | 최종 보고서 모델 |
| `mcp_servers` | 각각 | arxiv, tavily, serper | MCP 서버 매핑 |

### 2.3 SimpleMCPReActAgent

**경로**: `youngs75_a2a.agents.simple_react.agent`

```python
class SimpleMCPReActAgent(BaseGraphAgent):
    NODE_NAMES = {"REACT": "react_agent"}
```

**SimpleReActConfig** (`youngs75_a2a.agents.simple_react.config`):

| 필드 | 환경변수 | 기본값 | 설명 |
|------|----------|--------|------|
| `system_prompt` | - | 검색 전문가 프롬프트 | 시스템 프롬프트 |
| `mcp_servers` | `TAVILY_MCP_URL` | `{"tavily": "http://localhost:3001/mcp/"}` | MCP 서버 |

### 2.4 OrchestratorAgent

**경로**: `youngs75_a2a.agents.orchestrator.agent`

```python
class OrchestratorAgent(BaseGraphAgent):
    NODE_NAMES = {
        "CLASSIFY": "classify",
        "DELEGATE": "delegate",
        "RESPOND": "respond",
    }
```

**OrchestratorConfig** (`youngs75_a2a.agents.orchestrator.config`):

```python
class AgentEndpoint(BaseAgentConfig):
    name: str
    url: str
    description: str

class OrchestratorConfig(BaseAgentConfig):
    agent_endpoints: list[AgentEndpoint]

    def get_agent_descriptions(self) -> str: ...
    def get_endpoint_url(self, agent_name) -> str | None: ...
```

---

## 3. A2A 모듈 (`youngs75_a2a.a2a`)

> import 경로 주의: 패키지 내부에서는 `youngs75_a2a.a2a`로 접근하지만, 실제 디렉토리는 `a2a_local/`이다.

### 3.1 Executor

**경로**: `youngs75_a2a.a2a.executor`

```python
class BaseAgentExecutor(AgentExecutor):
    """일반 async callable을 A2A로 래핑."""
    def __init__(self, agent_fn: AgentFn, *, execution_timeout: float | None = None): ...
    async def execute(self, context, event_queue) -> None: ...
    async def cancel(self, context, event_queue) -> None: ...

class LGAgentExecutor(AgentExecutor):
    """LangGraph CompiledStateGraph를 A2A로 래핑."""
    def __init__(self, graph, result_extractor=None, *, execution_timeout=None): ...
    async def execute(self, context, event_queue) -> None: ...
    async def cancel(self, context, event_queue) -> None: ...
```

| 기능 | `BaseAgentExecutor` | `LGAgentExecutor` |
|------|-------|------|
| 입력 | `async callable` | `CompiledStateGraph` |
| 스트리밍 | `AsyncIterator[str]` 반환 시 자동 | `graph.astream()` |
| 취소 | `asyncio.Task.cancel()` | 폴링 + `asyncio.Task.cancel()` 하이브리드 |

### 3.2 Server

**경로**: `youngs75_a2a.a2a.server`

```python
def create_agent_card(
    name: str,
    description: str = "",
    url: str = "http://localhost:8080",
    skills: list[AgentSkill] | None = None,
    streaming: bool = True,
) -> AgentCard: ...

def build_app(executor, agent_card) -> A2AStarletteApplication: ...

def run_server(
    executor: AgentExecutor,
    name: str = "my-agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8080,
): ...
```

### 3.3 Discovery

**경로**: `youngs75_a2a.a2a.discovery`

```python
class AgentCardEntry:
    card: AgentCard
    url: str
    @property
    def is_healthy(self) -> bool: ...
    def mark_healthy(self) -> None: ...
    def mark_failed(self) -> None: ...

class DiscoveryResult:
    entry: AgentCardEntry
    match_score: float              # 0.0 ~ 1.0
    matched_skills: list[str]
    matched_tags: list[str]

class AgentCardRegistry:
    def register(self, card, url=None) -> AgentCardEntry: ...
    def unregister(self, url) -> bool: ...
    async def discover(self, url, *, timeout=10.0) -> AgentCardEntry | None: ...
    async def discover_many(self, urls, *, timeout=10.0) -> list[AgentCardEntry]: ...
    def get_by_name(self, name) -> AgentCardEntry | None: ...
    def list_all(self) -> list[AgentCardEntry]: ...
    def list_healthy(self) -> list[AgentCardEntry]: ...
    def find_by_skill(self, skill_query, *, only_healthy=True) -> list[DiscoveryResult]: ...
    def find_by_tags(self, tags, *, only_healthy=True) -> list[DiscoveryResult]: ...
    async def health_check(self, url) -> bool: ...
    async def health_check_all(self) -> dict[str, bool]: ...
    def start_periodic_health_check(self) -> None: ...
    def stop_periodic_health_check(self) -> None: ...
```

### 3.4 Router

**경로**: `youngs75_a2a.a2a.router`

```python
class RoutingMode(str, Enum):
    SKILL_BASED = "skill_based"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"

class RoutingDecision:
    target_url: str
    agent_name: str
    reason: str
    confidence: float
    alternatives: list[str]

class DelegationResult:
    success: bool
    agent_name: str
    agent_url: str
    response: SendMessageResponse | None
    streaming_response: AsyncStreamingResponse | None
    error: str | None
    latency_ms: float

class TaskDelegator:
    async def delegate(self, url, content, *, agent_name="unknown", fallback_urls=None) -> DelegationResult: ...
    async def delegate_streaming(self, url, content, *, agent_name="unknown") -> DelegationResult: ...
    async def delegate_parallel(self, targets, content) -> list[DelegationResult]: ...
    async def delegate_with_consensus(self, targets, content, *, min_success=1) -> list[DelegationResult]: ...

class AgentRouter:
    def register_agent(self, card, url=None) -> AgentCardEntry: ...
    async def discover(self, url) -> AgentCardEntry | None: ...
    def route(self, query, *, required_tags=None) -> RoutingDecision | None: ...
    async def delegate(self, decision, content) -> DelegationResult: ...
    async def route_and_delegate(self, query, *, required_tags=None) -> DelegationResult: ...
    async def route_and_delegate_streaming(self, query, *, required_tags=None) -> DelegationResult: ...
    async def broadcast(self, query, *, required_tags=None, max_agents=0) -> list[DelegationResult]: ...
```

### 3.5 Resilience

**경로**: `youngs75_a2a.a2a.resilience`

```python
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    def compute_delay(self, attempt) -> float: ...
    def is_retryable(self, error) -> bool: ...

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    @property
    def state(self) -> CircuitState: ...
    def can_execute(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    def reset(self) -> None: ...

class AgentMonitor:
    def get_stats(self, url) -> AgentHealthStats: ...
    def record_success(self, url, latency_ms) -> None: ...
    def record_failure(self, url, error) -> None: ...
    def get_healthy_urls(self, min_success_rate=0.5) -> list[str]: ...

class ResilientA2AClient:
    async def send_message(self, content, *, context_id=None, task_id=None) -> SendMessageResponse: ...
    async def send_message_streaming(self, content) -> AsyncStreamingResponse: ...
```

### 3.6 Streaming

**경로**: `youngs75_a2a.a2a.streaming`

```python
class StreamChunk:
    text: str
    accumulated: str
    state: str = "working"
    index: int = 0
    elapsed_ms: float = 0.0

class StreamingResponseCollector:
    def __init__(self, *, timeout=300.0, chunk_callback=None): ...
    async def collect_stream(self, url, content) -> AsyncIterator[StreamChunk]: ...
    def get_final_text(self) -> str: ...

async def stream_agent_response(url, content, *, timeout=300.0, on_chunk=None) -> str: ...
```

---

## 4. CLI 모듈 (`youngs75_a2a.cli`)

### 4.1 CLIConfig

**경로**: `youngs75_a2a.cli.config`

| 필드 | 환경변수 | 기본값 | 설명 |
|------|----------|--------|------|
| `default_agent` | `CLI_DEFAULT_AGENT` | `"coding_assistant"` | 기본 에이전트 |
| `stream_output` | - | `True` | 스트리밍 출력 |
| `history_file` | `CLI_HISTORY_FILE` | `".cli_history"` | 히스토리 파일 경로 |
| `max_history` | - | `1000` | 최대 히스토리 수 |
| `theme` | `CLI_THEME` | `"monokai"` | 테마 |
| `skills_dir` | `SKILLS_DIR` | `None` | 스킬 디렉토리 |
| `checkpointer_backend` | `CLI_CHECKPOINTER` | `"memory"` | 체크포인터 (memory/sqlite) |
| `checkpointer_sqlite_path` | `CLI_CHECKPOINTER_SQLITE_PATH` | `".checkpoints.db"` | SQLite 경로 |
| `langfuse_enabled` | `CLI_LANGFUSE_ENABLED` | `True` | Langfuse 관측성 활성화 |

### 4.2 슬래시 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/help`, `/h` | 도움말 |
| `/agent <name>` | 에이전트 전환 |
| `/agents` | 사용 가능한 에이전트 목록 |
| `/skill list` | 등록된 스킬 목록 |
| `/skill activate <name>` | 스킬 활성화 (L2 로드) |
| `/history` | 최근 대화 기록 표시 |
| `/history clear` | 대화 기록 초기화 |
| `/eval` | 에이전트 평가 실행 (DeepEval) |
| `/eval status` | 마지막 평가 결과 요약 |
| `/eval remediate` | Remediation 실행 (Loop 3) |
| `/eval remediate status` | 마지막 Remediation 결과 |
| `/clear` | 대화 기록 초기화 |
| `/session` | 현재 세션 정보 |
| `/memory` | 메모리 상태 |
| `/quit`, `/exit`, `/q` | 종료 |

### 4.3 CLISession

**경로**: `youngs75_a2a.cli.session`

```python
class CLISession:
    info: SessionInfo
    memory: MemoryStore
    skills: SkillRegistry
    checkpointer: Any

    def add_message(self, role, content) -> None: ...
    def clear_history(self) -> None: ...
    def get_cached_agent(self, name) -> Any | None: ...
    def cache_agent(self, name, agent) -> None: ...
    def switch_agent(self, agent_name) -> None: ...
    def get_history_summary(self, limit=10) -> list[dict]: ...
    def activate_skill(self, name) -> str | None: ...
```

### 4.4 CLIRenderer

**경로**: `youngs75_a2a.cli.renderer`

Rich 콘솔 기반 출력 렌더러. 마크다운 렌더링, 토큰 스트리밍, 시스템/에러 메시지를 지원한다.

### 4.5 진입점

```python
# pyproject.toml [project.scripts]
youngs75-agent = "youngs75_a2a.cli.app:run_cli"

# 또는 모듈 실행
python -m youngs75_a2a.cli
```

---

## 5. 평가 파이프라인 (`youngs75_a2a.eval_pipeline`)

### 5.1 Settings

**경로**: `youngs75_a2a.eval_pipeline.settings`

| 필드 | 환경변수 | 기본값 | 설명 |
|------|----------|--------|------|
| `env` | `ENV` | `"local"` | 런타임 환경 |
| `service_name` | `SERVICE_NAME` | `"youngs75-a2a"` | 서비스명 |
| `app_version` | `APP_VERSION` | `"0.1.0"` | 앱 버전 |
| `openai_model_name` | `OPENAI_MODEL_NAME` | `"gpt-4o-mini"` | OpenAI 모델 |
| `openrouter_model_name` | `DEFAULT_MODEL` | `"gpt-4o-mini"` | OpenRouter 모델 |
| `openrouter_api_key` | `OPENROUTER_API_KEY` | `""` | OpenRouter API 키 |
| `langfuse_host` | `LANGFUSE_HOST` | `""` | Langfuse 호스트 |
| `langfuse_public_key` | `LANGFUSE_PUBLIC_KEY` | `""` | Langfuse Public Key |
| `langfuse_secret_key` | `LANGFUSE_SECRET_KEY` | `""` | Langfuse Secret Key |
| `langfuse_tracing_enabled` | `LANGFUSE_TRACING_ENABLED` | `True` | 트레이싱 활성화 |
| `langfuse_sample_rate` | `LANGFUSE_SAMPLE_RATE` | `1.0` | 샘플링 비율 |
| `data_dir` | `DATA_DIR` | `./data` | 데이터 디렉토리 |
| `local_corpus_dir` | `LOCAL_CORPUS_DIR` | `./data/corpus` | 로컬 코퍼스 |

### 5.2 Loop 1 — Dataset 생성

**경로**: `youngs75_a2a.eval_pipeline.loop1_dataset`

| 모듈 | 역할 |
|------|------|
| `synthesizer` | 합성 데이터 생성 |
| `golden_builder` | Golden Dataset 빌드 |
| `feedback_augmenter` | 피드백 증강 |
| `csv_exporter` / `csv_importer` | CSV 내보내기/가져오기 |

### 5.3 Loop 2 — 메트릭 평가

**경로**: `youngs75_a2a.eval_pipeline.loop2_evaluation`

| 모듈 | 역할 |
|------|------|
| `metrics_registry` | 메트릭 레지스트리 (RAG 4종 + Agent 2종 + Custom 7종 = 13개) |
| `batch_evaluator` | 오프라인/온라인 배치 평가 |
| `langfuse_bridge` | Langfuse 트레이스 fetch/push |
| `agent_metrics` | 에이전트 전용 메트릭 |
| `rag_metrics` | RAG 전용 메트릭 |
| `custom_metrics` | 커스텀 메트릭 |
| `calibration_cases` | 교정 데이터 |
| `prompt_optimizer` | 프롬프트 최적화 |

### 5.4 Loop 3 — Remediation

**경로**: `youngs75_a2a.eval_pipeline.loop3_remediation`

| 모듈 | 역할 |
|------|------|
| `remediation_agent` | 개선 에이전트 (분석 -> 최적화 -> 추천) |
| `recommendation` | `RecommendationReport` 스키마 |
| `analysis_tools` | 평가 결과 읽기 도구 |

```python
async def run_remediation(
    *,
    eval_results_dir: str | None = None,
    thread_id: str = "remediation",
) -> RecommendationReport: ...
```

### 5.5 관측성 (Observability)

**경로**: `youngs75_a2a.eval_pipeline.observability`

| 모듈 | 역할 |
|------|------|
| `langfuse` | Langfuse 클라이언트 초기화 |
| `callback_handler` | LangGraph 콜백 핸들러 (트레이스/메트릭 수집) |

```python
def create_langfuse_handler() -> Any | None: ...
def build_observed_config(handler, session_id, thread_id, agent_name) -> RunnableConfig: ...
def safe_flush() -> None: ...

class AgentMetricsCollector:
    def __init__(self, agent_name): ...
    def record_node_start(self, node) -> None: ...
    def record_node_end(self, node) -> None: ...
    def record_llm_tokens(self, prompt_tokens, completion_tokens) -> None: ...
    def record_error(self) -> None: ...
    def finalize(self) -> None: ...
    def to_dict(self) -> dict: ...
```

---

## 6. MCP 서버 (`youngs75_a2a.mcp_servers`)

### 6.1 Code Tools MCP 서버

**경로**: `youngs75_a2a.mcp_servers.code_tools.server`  
**포트**: 3003 (환경변수 `CODE_TOOLS_PORT`)  
**워크스페이스**: 현재 디렉토리 (환경변수 `CODE_TOOLS_WORKSPACE`)

| 도구 | 설명 |
|------|------|
| `read_file(path, max_lines=500)` | 프로젝트 파일 읽기 (라인 번호 포함) |
| `write_file(path, content)` | 프로젝트 파일 쓰기 |
| `list_directory(path, max_depth=2)` | 디렉토리 트리 출력 |
| `search_code(pattern, path=".", max_results=20)` | 코드 검색 (grep) |
| `run_python(code, timeout=30)` | Python 코드 실행 (샌드박스) |

모든 파일 접근은 워크스페이스 내부로 제한된다 (`_safe_path()` 검증).

---

## 7. 환경변수 종합

### 7.1 LLM 모델

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PROVIDER` | `openai` | 모델 프로바이더 |
| `MODEL_NAME` | `gpt-5.4` | 기본 모델명 |
| `TEMPERATURE` | `0.1` | LLM 온도 |
| `STRONG_MODEL` | `gpt-5.4` | Strong 티어 모델 |
| `STRONG_PROVIDER` | `openai` | Strong 티어 프로바이더 |
| `STRONG_CONTEXT_WINDOW` | `128000` | Strong 티어 컨텍스트 |
| `DEFAULT_MODEL` | `gpt-5.4` | Default 티어 모델 |
| `DEFAULT_PROVIDER` | `openai` | Default 티어 프로바이더 |
| `DEFAULT_CONTEXT_WINDOW` | `128000` | Default 티어 컨텍스트 |
| `FAST_MODEL` | `gpt-4.1-mini` | Fast 티어 모델 |
| `FAST_PROVIDER` | `openai` | Fast 티어 프로바이더 |
| `FAST_CONTEXT_WINDOW` | `128000` | Fast 티어 컨텍스트 |
| `PURPOSE_TIERS` | (기본 매핑) | JSON: purpose -> tier 오버라이��� |
| `CODING_GEN_MODEL` | (미설정) | Coding 생성 모델 오버라이드 |
| `CODING_VERIFY_MODEL` | (미설정) | Coding 검증 모델 오버라이드 |
| `OPENAI_API_KEY` | - | OpenAI API 키 |
| `OPENROUTER_API_KEY` | - | OpenRouter API 키 |

### 7.2 MCP 서버

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TAVILY_MCP_URL` | `http://localhost:3001/mcp/` | Tavily MCP 엔드포인트 |
| `ARXIV_MCP_URL` | `http://localhost:3000/mcp/` | arXiv MCP 엔드포인트 |
| `SERPER_MCP_URL` | `http://localhost:3002/mcp/` | Serper MCP 엔드포인트 |
| `CODE_TOOLS_MCP_URL` | `http://localhost:3003/mcp/` | Code Tools MCP 엔드포인트 |
| `CODE_TOOLS_PORT` | `3003` | Code Tools 서버 포트 |
| `CODE_TOOLS_WORKSPACE` | `os.getcwd()` | Code Tools 워크스페이스 |

### 7.3 에이전트 실행

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALLOW_CLARIFICATION` | `true` | DeepResearch 질문 명확화 |
| `MAX_CONCURRENT_RESEARCH` | `3` | 병렬 연구 동시성 |
| `MAX_RESEARCHER_ITERATIONS` | `3` | 연구자 최대 반복 |
| `ENABLE_HITL` | `false` | HITL 승인 루프 |

### 7.4 CLI

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CLI_DEFAULT_AGENT` | `coding_assistant` | 기본 에이전트 |
| `CLI_HISTORY_FILE` | `.cli_history` | 히스토리 파일 |
| `CLI_THEME` | `monokai` | 테마 |
| `SKILLS_DIR` | (미설정) | 스킬 디렉토리 |
| `CLI_CHECKPOINTER` | `memory` | 체크포인터 백엔드 |
| `CLI_CHECKPOINTER_SQLITE_PATH` | `.checkpoints.db` | SQLite 경로 |
| `CLI_LANGFUSE_ENABLED` | `1` | Langfuse 활성화 |

### 7.5 Langfuse (관측성)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LANGFUSE_HOST` | - | Langfuse 서버 URL |
| `LANGFUSE_PUBLIC_KEY` | - | Langfuse Public Key |
| `LANGFUSE_SECRET_KEY` | - | Langfuse Secret Key |
| `LANGFUSE_TRACING_ENABLED` | `1` | 트레이싱 활성화 |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | 샘플링 비율 |

### 7.6 애플리케이션

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ENV` | `local` | 런타임 환경 |
| `SERVICE_NAME` | `youngs75-coding-ai-agent` | 서비스명 |
| `APP_VERSION` | `0.1.0` | 앱 버전 |
| `DATA_DIR` | `./data` | 데이터 디렉토리 |
| `LOCAL_CORPUS_DIR` | `./data/corpus` | 로컬 코퍼스 |
