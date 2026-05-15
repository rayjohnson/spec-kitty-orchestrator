"""Tests for provider agent invokers."""

from __future__ import annotations

from pathlib import Path

from spec_kitty_orchestrator.agents import all_agent_ids, get_invoker
from spec_kitty_orchestrator.agents.letta import LettaInvoker
from spec_kitty_orchestrator.agents.pi import PiInvoker


def test_registry_includes_pi_and_letta() -> None:
    assert "pi" in all_agent_ids()
    assert "letta" in all_agent_ids()
    assert isinstance(get_invoker("pi"), PiInvoker)
    assert isinstance(get_invoker("letta"), LettaInvoker)


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
