"""langfuse_trace_exporter 유닛 테스트."""

from datetime import datetime, timezone

from coding_agent.utils.langfuse_trace_exporter import (
    Message,
    Generation,
    TraceConversation,
    format_conversation_markdown,
    _parse_message,
    _extract_messages_from_input,
    _extract_output_message,
    _extract_user_request,
    _extract_final_output,
    _format_tool_calls,
)


class TestParseMessage:
    def test_basic_message(self):
        msg = _parse_message({"role": "user", "content": "hello"})
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls == []

    def test_none_content(self):
        msg = _parse_message({"role": "assistant", "content": None})
        assert msg.content == ""

    def test_multimodal_content(self):
        msg = _parse_message({
            "role": "user",
            "content": [{"text": "part1"}, {"text": "part2"}],
        })
        assert "part1" in msg.content
        assert "part2" in msg.content

    def test_tool_calls(self):
        msg = _parse_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        })
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_file"

    def test_tool_call_id(self):
        msg = _parse_message({
            "role": "tool",
            "content": "result",
            "tool_call_id": "tc1",
        })
        assert msg.tool_call_id == "tc1"


class TestExtractMessages:
    def test_from_dict_with_messages_key(self):
        inp = {"messages": [{"role": "user", "content": "hi"}]}
        msgs = _extract_messages_from_input(inp)
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_from_list(self):
        inp = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        msgs = _extract_messages_from_input(inp)
        assert len(msgs) == 2

    def test_none_input(self):
        assert _extract_messages_from_input(None) == []


class TestExtractOutput:
    def test_dict_output(self):
        out = {"role": "assistant", "content": "response"}
        msg = _extract_output_message(out)
        assert msg is not None
        assert msg.content == "response"

    def test_string_output(self):
        msg = _extract_output_message("plain text")
        assert msg is not None
        assert msg.content == "plain text"

    def test_none_output(self):
        assert _extract_output_message(None) is None


class TestExtractUserRequest:
    def test_from_trace_output_messages(self):
        output = {
            "messages": [
                {"type": "human", "content": "make a todo app"},
                {"type": "ai", "content": "ok"},
            ]
        }
        assert _extract_user_request(None, output) == "make a todo app"

    def test_from_string_input(self):
        assert _extract_user_request("build something", None) == "build something"

    def test_none(self):
        assert _extract_user_request(None, None) is None


class TestExtractFinalOutput:
    def test_from_messages(self):
        output = {
            "messages": [
                {"type": "human", "content": "request"},
                {"type": "ai", "content": "first response"},
                {"type": "ai", "content": "final response"},
            ]
        }
        assert _extract_final_output(output) == "final response"

    def test_from_string(self):
        assert _extract_final_output("done") == "done"


class TestFormatToolCalls:
    def test_basic(self):
        result = _format_tool_calls([{"name": "read_file", "arguments": '{"path": "x.py"}'}])
        assert "read_file" in result
        assert "x.py" in result

    def test_truncation(self):
        long_args = '{"content": "' + "x" * 1000 + '"}'
        result = _format_tool_calls([{"name": "write_file", "arguments": long_args}])
        assert "truncated" in result


class TestFormatConversationMarkdown:
    def _make_conversation(self) -> TraceConversation:
        return TraceConversation(
            trace_id="test-trace-123",
            trace_name="OrchestratorAgent",
            session_id="test-session",
            timestamp=datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc),
            user_input="make a todo app",
            agent_output="Here is your todo app...",
            total_cost=0.005,
            generations=[
                Generation(
                    observation_id="gen1",
                    model="qwen3-max",
                    parent_name="classify",
                    start_time=datetime(2026, 4, 9, 12, 0, 1, tzinfo=timezone.utc),
                    input_messages=[
                        Message(role="system", content="You are a router."),
                        Message(role="user", content="make a todo app"),
                    ],
                    output_message=Message(role="assistant", content="coding_assistant"),
                    usage={"input": 100, "output": 10},
                    latency=1.2,
                ),
                Generation(
                    observation_id="gen2",
                    model="qwen3-coder",
                    parent_name="generate_code",
                    start_time=datetime(2026, 4, 9, 12, 0, 5, tzinfo=timezone.utc),
                    input_messages=[
                        Message(role="system", content="You are a coder."),
                        Message(role="user", content="implement todo app"),
                    ],
                    output_message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[{"name": "write_file", "arguments": '{"path": "app.py"}'}],
                    ),
                    usage={"input": 500, "output": 200},
                    latency=8.5,
                ),
            ],
        )

    def test_basic_format(self):
        conv = self._make_conversation()
        result = format_conversation_markdown([conv])

        assert "# Langfuse Trace Export" in result
        assert "test-session" in result
        assert "OrchestratorAgent" in result
        assert "make a todo app" in result
        assert "qwen3-max" in result
        assert "qwen3-coder" in result
        assert "coding_assistant" in result
        assert "Step 1" in result
        assert "Step 2" in result

    def test_verbose_mode(self):
        conv = self._make_conversation()
        result = format_conversation_markdown([conv], verbose=True)
        # verbose에서는 system 전체 내용이 표시
        assert "You are a router." in result
        assert "You are a coder." in result

    def test_empty_list(self):
        result = format_conversation_markdown([])
        assert result == ""

    def test_cost_display(self):
        conv = self._make_conversation()
        result = format_conversation_markdown([conv])
        assert "$0.005" in result
