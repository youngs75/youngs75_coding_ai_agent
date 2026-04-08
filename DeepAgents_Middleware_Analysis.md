# DeepAgents Middleware System Analysis

## 1. Middleware Base Protocol

**Location:** `libs/deepagents/deepagents/middleware/__init__.py` (lines 1-48)

### Interface Definition

The middleware system is built on **two paths**:

1. **SDK middleware** (this package) — injected automatically
2. **Consumer-provided tools** — passed via `tools` parameter

### Why Middleware Instead of Plain Tools?

Middleware subclasses `AgentMiddleware` and overrides **`wrap_model_call()`** hook to:
- **Filter tools dynamically** — e.g., remove `execute` tool if backend doesn't support it
- **Inject system-prompt context** — e.g., `MemoryMiddleware` injects AGENTS.md on every call
- **Transform messages** — e.g., `SummarizationMiddleware` counts tokens and truncates history
- **Maintain cross-turn state** — middleware reads/writes typed state dict persisting across turns

Plain tools cannot do this — they're only invoked **by** the LLM, not **before**.

### Core Methods (from LangChain's `AgentMiddleware`):

```python
wrap_model_call(request: ModelRequest, handler: Callable) -> ModelResponse
awrap_model_call(request: ModelRequest, handler: Callable) -> Awaitable[ModelResponse]
```

- **`request`** — encapsulates system message, messages, tools, state
- **`handler`** — next middleware in the chain (or the model call itself)
- Must call `handler(request)` or async `await handler(request)` to proceed

---

## 2. Middleware Chaining (Onion Pattern)

**Location:** `libs/deepagents/deepagents/graph.py` lines 363-395

### Construction in `create_deep_agent()`

```python
# Main agent middleware stack (lines 363-395)
deepagent_middleware: list[AgentMiddleware] = [
    TodoListMiddleware(),
]
if skills is not None:
    deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
deepagent_middleware.extend([
    FilesystemMiddleware(backend=backend),
    SubAgentMiddleware(backend=backend, subagents=inline_subagents),
    create_summarization_middleware(model, backend),
    PatchToolCallsMiddleware(),
])

if async_subagents:
    deepagent_middleware.append(AsyncSubAgentMiddleware(async_subagents=async_subagents))

if middleware:
    deepagent_middleware.extend(middleware)  # User middleware inserted here

deepagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
if memory is not None:
    deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
if interrupt_on is not None:
    deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
```

### Execution Order

**Base stack → User middleware → Tail stack**

Each middleware in the list wraps the next one. When a model call happens:

1. **Request enters** first middleware in list
2. Each middleware can **modify request** before calling `handler()`
3. `handler()` calls next middleware in chain (or model)
4. **Response returns** through the chain in reverse

This is the **onion pattern**: outer layers execute first on request, last on response.

---

## 3. ModelRequest / ModelResponse Types

**Location:** `libs/deepagents/deepagents/middleware/subagents.py` lines 8, 522-524

Imported from `langchain.agents.middleware.types`:

```python
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
```

### ModelRequest Fields

From `wrap_model_call` signatures and usage:

- **`system_message`** — `SystemMessage | None` — the system prompt
- **`messages`** — `list[AnyMessage]` — conversation history
- **`tools`** — optional list of available tools
- **`state`** — current agent state (TypedDict)
- **`runtime`** — runtime context with stream writer, store, context

**Methods:**
- `request.override(system_message=..., messages=..., ...)` — creates modified copy

### ModelResponse

- Generic type `ModelResponse[ResponseT]`
- Contains model's output (typically tool calls + text)
- Can be extended to `ExtendedModelResponse` with state updates (see summarization)

---

## 4. SummarizationMiddleware Implementation

**Location:** `libs/deepagents/deepagents/middleware/summarization.py`

### Key Concept: 2-Phase Strategy

**Phase 1 (Lightweight):** Truncate large tool-call arguments (line 674-733)
- Triggered at *lower* token threshold (lines 567-595)
- Only truncates `args` in `AIMessage.tool_calls` for old messages
- Keeps recent messages intact

**Phase 2 (Heavy):** Full conversation compaction (lines 885-987)
- Triggered when token count exceeds threshold
- Partitions messages into "to summarize" and "to keep"
- Summarizes old messages via LLM
- Offloads full history to backend
- Replaces evicted messages with summary

### `before_model` Hook Implementation

**Location:** `wrap_model_call()` (lines 885-987)

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | ExtendedModelResponse:
    # Step 1: Get effective messages (apply previous summarization events)
    effective_messages = self._get_effective_messages(request)  # line 921
    
    # Step 2: Truncate large args if configured
    truncated_messages, _ = self._truncate_args(
        effective_messages,
        request.system_message,
        request.tools,
    )  # lines 924-928
    
    # Step 3: Check if summarization threshold met
    total_tokens = self.token_counter(counted_messages, tools=request.tools)
    should_summarize = self._should_summarize(truncated_messages, total_tokens)  # line 936
    
    # If no summarization needed, return with truncated messages
    if not should_summarize:
        try:
            return handler(request.override(messages=truncated_messages))  # line 941
        except ContextOverflowError:
            pass  # Fallback to summarization on context overflow (line 942)
    
    # Step 4: Perform summarization
    cutoff_index = self._determine_cutoff_index(truncated_messages)  # line 947
    messages_to_summarize, preserved_messages = self._partition_messages(
        truncated_messages, cutoff_index
    )  # line 952
    
    # Step 5: Offload to backend BEFORE summarization
    backend = self._get_backend(request.state, request.runtime)
    file_path = self._offload_to_backend(backend, messages_to_summarize)  # line 957
    
    # Step 6: Generate summary
    summary = self._create_summary(messages_to_summarize)  # line 964
    
    # Step 7: Build summary message with file path reference
    new_messages = self._build_new_messages_with_path(summary, file_path)  # line 967
    
    # Step 8: Create summarization event for state
    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff_index,
        "summary_message": new_messages[0],
        "file_path": file_path,
    }  # lines 973-977
    
    # Step 9: Return with ExtendedModelResponse to update state
    modified_messages = [*new_messages, *preserved_messages]
    response = handler(request.override(messages=modified_messages))
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"_summarization_event": new_event}),
    )  # lines 984-987
```

### Cutoff Index Determination

**`_determine_cutoff_index()`** (delegated to LangChain helper, line 331-333)

Uses the **`keep`** configuration to decide how many recent messages to preserve:
- **`("messages", N)`** — keep last N messages
- **`("tokens", N)`** — keep messages up to N tokens of context
- **`("fraction", F)`** — keep last F% of context window

### Truncation Strategy

**`_truncate_args()`** (lines 674-733)

For each message before cutoff index, if it's an `AIMessage` with `tool_calls`:
1. Check if tool name is in `{"write_file", "edit_file"}` (line 714)
2. If arg value length > `max_length`, truncate: `value[:20] + truncation_text` (line 662)
3. Return modified message copy

This is **lightweight** and fires before full summarization.

### Offloading to Backend

**`_offload_to_backend()`** (lines 735-807)

1. Filter out previous summary messages (line 761)
2. Format messages as markdown (line 764)
3. Read existing history file (lines 769-781)
4. Append new section with timestamp header (line 783)
5. Write back to backend (line 786)
6. Return file path or None on error

Path format: `{artifacts_root}/conversation_history/{thread_id}.md` (line 417)

---

## 5. SubAgentMiddleware Implementation

**Location:** `libs/deepagents/deepagents/middleware/subagents.py` lines 392-540

### Dynamic Task Tool Injection

**`__init__()`** (lines 440-467):

```python
def __init__(
    self,
    *,
    backend: BackendProtocol | BackendFactory,
    subagents: Sequence[SubAgent | CompiledSubAgent],
    system_prompt: str | None = TASK_SYSTEM_PROMPT,
    task_description: str | None = None,
) -> None:
    self._subagents = subagents
    subagent_specs = self._get_subagents()  # Compile all subagent runnables
    
    task_tool = _build_task_tool(subagent_specs, task_description)  # line 458
    
    # Build system prompt with available agents
    if system_prompt and subagent_specs:
        agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
        self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
    
    self.tools = [task_tool]  # line 467
```

### System Prompt Injection

**`wrap_model_call()`** (lines 520-529):

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT]:
    if self.system_prompt is not None:
        new_system_message = append_to_system_message(
            request.system_message, self.system_prompt
        )
        return handler(request.override(system_message=new_system_message))
    return handler(request)
```

This appends the task tool instructions and list of available subagents to the system message on every call.

### Task Tool Implementation

**`_build_task_tool()`** (lines 298-389):

Creates a `StructuredTool` with:
- **`description`** — includes list of available agents (line 318)
- **`args_schema`** — `TaskToolSchema` with `description` and `subagent_type` fields
- **`func`** — synchronous `task()` function (lines 352-365)
- **`coroutine`** — async `atask()` function (lines 367-380)

#### Task Function Logic

```python
def task(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,
) -> str | Command:
    # Validate subagent exists
    if subagent_type not in subagent_graphs:
        return f"We cannot invoke subagent {subagent_type} because it does not exist..."
    
    # Validate tool call ID
    if not runtime.tool_call_id:
        raise ValueError("Tool call ID is required for subagent invocation")
    
    # Prepare state for subagent
    subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
    
    # Invoke subagent
    result = subagent.invoke(subagent_state)  # line 364
    
    # Return Command with state update
    return _return_command_with_state_update(result, runtime.tool_call_id)
```

#### State Preparation

**`_validate_and_prepare_state()`** (lines 344-350):

```python
def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime):
    subagent = subagent_graphs[subagent_type]
    
    # Create new state dict, excluding certain keys
    subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
    
    # Set human message as the task description
    subagent_state["messages"] = [HumanMessage(content=description)]
    
    return subagent, subagent_state
```

**Excluded state keys** (line 126):
```python
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata", "memory_contents"}
```

These prevent parent state from leaking into child agents.

#### Result Processing

**`_return_command_with_state_update()`** (lines 324-342):

```python
def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
    # Validate result has 'messages' key
    if "messages" not in result:
        raise ValueError("CompiledSubAgent must return a state containing a 'messages' key")
    
    # Extract non-excluded state keys for merging back
    state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
    
    # Extract final message text
    message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
    
    # Return Command to update parent state
    return Command(
        update={
            **state_update,
            "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
        }
    )
```

---

## 6. MemoryMiddleware Implementation

**Location:** `libs/deepagents/deepagents/middleware/memory.py`

### State Schema

**`MemoryState`** (lines 80-88):

```python
class MemoryState(AgentState):
    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
```

The `PrivateStateAttr` annotation marks `memory_contents` as **not included in final agent state** — it's only used internally.

### Loading Memory Before Agent

**`before_agent()`** (lines 238-270):

```python
def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:
    # Skip if already loaded
    if "memory_contents" in state:
        return None
    
    # Resolve backend
    backend = self._get_backend(state, runtime, config)
    contents: dict[str, str] = {}
    
    # Download all source files
    results = backend.download_files(list(self.sources))
    for path, response in zip(self.sources, results, strict=True):
        if response.error is not None:
            if response.error == "file_not_found":
                continue  # Skip missing files
            raise ValueError(f"Failed to download {path}: {response.error}")
        if response.content is not None:
            contents[path] = response.content.decode("utf-8")
    
    return MemoryStateUpdate(memory_contents=contents)
```

Called **once per agent run** — skips if already loaded (line 253).

### Injecting Memory into System Prompt

**`wrap_model_call()`** (lines 322-337):

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT]:
    modified_request = self.modify_request(request)
    return handler(modified_request)

def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    contents = request.state.get("memory_contents", {})
    agent_memory = self._format_agent_memory(contents)
    new_system_message = append_to_system_message(request.system_message, agent_memory)
    return request.override(system_message=new_system_message)
```

### Memory Formatting

**`_format_agent_memory()`** (lines 218-236):

```python
def _format_agent_memory(self, contents: dict[str, str]) -> str:
    if not contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    
    sections = [f"{path}\n{contents[path]}" for path in self.sources if contents.get(path)]
    
    if not sections:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    
    memory_body = "\n\n".join(sections)
    return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)
```

Wraps memory in `MEMORY_SYSTEM_PROMPT` template (lines 97-156) with instructions on:
- How to learn from feedback
- When to update memories
- How to ask for information

---

## 7. State Extensions by Middleware

### SummarizationState (lines 151-157)

```python
class SummarizationState(AgentState):
    _summarization_event: Annotated[NotRequired[SummarizationEvent | None], PrivateStateAttr]
```

**SummarizationEvent structure** (lines 107-118):

```python
class SummarizationEvent(TypedDict):
    cutoff_index: int  # Where in messages list summarization occurred
    summary_message: HumanMessage  # The generated summary
    file_path: str | None  # Path to offloaded history in backend
```

Persists across turns so subsequent calls can apply previous summaries.

### MemoryState (lines 80-88)

```python
class MemoryState(AgentState):
    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
```

Stores loaded memory file contents.

### Usage in wrap_model_call

Middlewares access state via `request.state`:
```python
event = request.state.get("_summarization_event")
contents = request.state.get("memory_contents", {})
```

---

## 8. The `append_to_system_message` Utility

**Location:** `libs/deepagents/deepagents/middleware/_utils.py` lines 6-23

```python
def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.
    
    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.
    
    Returns:
        New SystemMessage with the text appended.
    """
    # Get existing content blocks or empty list
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    
    # Add separator if content already exists
    if new_content:
        text = f"\n\n{text}"
    
    # Append new text block
    new_content.append({"type": "text", "text": text})
    
    # Return new SystemMessage (immutable)
    return SystemMessage(content_blocks=new_content)
```

### Key Details

- **Handles None gracefully** — if no system message exists, creates new one with just the text
- **Adds separator** — inserts `\n\n` before appended text if content already exists (line 21)
- **Content block format** — LangChain's `SystemMessage` uses `content_blocks` (list of dicts) to support multimodal content
- **Immutable pattern** — returns new `SystemMessage` instead of mutating

Used by:
- `SubAgentMiddleware.wrap_model_call()` — appends task tool instructions
- `MemoryMiddleware.modify_request()` — appends loaded memory
- `graph.py` — combines custom system prompt with base agent prompt (line 404)

---

## Summary: Adaptation Points for Your Project

1. **Inherit from `AgentMiddleware`** and implement `wrap_model_call()` and `awrap_model_call()`
2. **Intercept `ModelRequest`** — modify `system_message`, `messages`, or `tools` before calling `handler()`
3. **Track state via TypedDict** — extend `AgentState` with `PrivateStateAttr` for internal fields
4. **Use `append_to_system_message()`** for safe system prompt modification
5. **Return `ModelResponse`** or `ExtendedModelResponse` with state updates via `Command`
6. **Chain in `create_agent()`'s `middleware` parameter** — order matters (onion pattern)
7. **Delegate to `request.override()`** for immutable request mutations

This pattern is composable — each middleware focuses on one concern and passes control via `handler()`.
