"""Tests for provider agent invokers."""

from __future__ import annotations

from pathlib import Path

from spec_kitty_orchestrator.agents import all_agent_ids, get_invoker
from spec_kitty_orchestrator.agents.claude import ClaudeInvoker
from spec_kitty_orchestrator.agents.codex import CodexInvoker
from spec_kitty_orchestrator.agents.letta import LettaInvoker
from spec_kitty_orchestrator.agents.opencode import OpenCodeInvoker
from spec_kitty_orchestrator.agents.pi import PiInvoker


def test_registry_includes_pi_and_letta() -> None:
    assert "pi" in all_agent_ids()
    assert "letta" in all_agent_ids()
    assert isinstance(get_invoker("pi"), PiInvoker)
    assert isinstance(get_invoker("letta"), LettaInvoker)


def test_registry_includes_primary_e2e_targets() -> None:
    assert isinstance(get_invoker("claude-code"), ClaudeInvoker)
    assert isinstance(get_invoker("codex"), CodexInvoker)
    assert isinstance(get_invoker("opencode"), OpenCodeInvoker)


def test_claude_builds_role_specific_headless_commands(tmp_path: Path) -> None:
    invoker = ClaudeInvoker()

    impl = invoker.build_command("prompt", tmp_path, "implementation")
    review = invoker.build_command("prompt", tmp_path, "review")

    assert impl[:3] == ["claude", "-p", "--output-format"]
    assert "--dangerously-skip-permissions" in impl
    assert impl[-2:] == ["--allowedTools", "Read,Write,Edit,Bash,Glob,Grep,TodoWrite"]
    assert review[-2:] == ["--allowedTools", "Read,Glob,Grep,Bash"]


def test_codex_builds_full_auto_json_stdin_command(tmp_path: Path) -> None:
    cmd = CodexInvoker().build_command("prompt", tmp_path, "implementation")

    assert cmd == ["codex", "exec", "-", "--json", "--full-auto"]


def test_opencode_builds_json_streaming_command(tmp_path: Path) -> None:
    cmd = OpenCodeInvoker().build_command("prompt", tmp_path, "implementation")

    assert cmd == ["opencode", "run", "--agent", "build", "--format", "json"]


def test_claude_parses_nested_json_result() -> None:
    result = ClaudeInvoker().parse_output(
        '{"result":{"files_modified":["src/app.py"],"commits":["abc123"]}}',
        "",
        0,
        0.5,
    )

    assert result.success is True
    assert result.files_modified == ["src/app.py"]
    assert result.commits_made == ["abc123"]


def test_codex_parses_json_result() -> None:
    result = CodexInvoker().parse_output(
        '{"files_modified":["src/app.py"],"commits":["abc123"]}',
        "",
        0,
        0.5,
    )

    assert result.success is True
    assert result.files_modified == ["src/app.py"]
    assert result.commits_made == ["abc123"]


def test_opencode_parses_jsonl_file_write_and_error_events() -> None:
    result = OpenCodeInvoker().parse_output(
        '{"type":"file_write","path":"src/app.py"}\n'
        '{"type":"error","message":"review failed"}\n',
        "",
        1,
        0.5,
    )

    assert result.success is False
    assert result.files_modified == ["src/app.py"]
    assert result.errors == ["review failed"]


def test_opencode_parses_nested_error_events() -> None:
    result = OpenCodeInvoker().parse_output(
        '{"type":"error","error":{"name":"UnknownError","data":{"message":"Model not found: test/model"}}}\n',
        "",
        1,
        0.5,
    )

    assert result.success is False
    assert result.errors == ["Model not found: test/model"]


def test_pi_builds_headless_json_commands(tmp_path: Path) -> None:
    invoker = PiInvoker()

    impl = invoker.build_command("prompt", tmp_path, "implementation")
    review = invoker.build_command("prompt", tmp_path, "review")

    assert impl[:4] == ["pi", "-p", "--mode", "json"]
    assert "--no-session" in impl
    assert impl[-2:] == ["--tools", "read,bash,edit,write,grep,find,ls"]
    assert review[-2:] == ["--tools", "read,grep,find,ls"]


def test_letta_builds_headless_json_commands(tmp_path: Path) -> None:
    invoker = LettaInvoker()

    impl = invoker.build_command("prompt", tmp_path, "implementation")
    review = invoker.build_command("prompt", tmp_path, "review")

    assert impl[:4] == ["letta", "-p", "--output-format", "json"]
    assert "--yolo" in impl
    assert review[-4:] == ["--permission-mode", "plan", "--tools", "Read,Glob,Grep"]


def test_pi_parses_jsonl_final_event() -> None:
    result = PiInvoker().parse_output(
        '{"type":"message","content":"working"}\n'
        '{"files_modified":["src/app.py"],"commits":["abc123"]}\n',
        "",
        0,
        1.2,
    )

    assert result.success is True
    assert result.files_modified == ["src/app.py"]
    assert result.commits_made == ["abc123"]


def test_letta_parses_json_result() -> None:
    result = LettaInvoker().parse_output(
        '{"result":"done","files_modified":["README.md"],"conversation_id":"c1"}',
        "",
        0,
        1.2,
    )

    assert result.success is True
    assert result.files_modified == ["README.md"]
