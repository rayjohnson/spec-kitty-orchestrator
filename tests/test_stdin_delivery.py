"""Concurrent stdin delivery for agent spawns.

Regression for: when several stdin-based agents (e.g. `claude -p`) are spawned in
the same window, the first-spawned child could read an empty/closed stdin before
the prompt was written and abort ("Input must be provided ... when using
--print"). The fix delivers each prompt via a pre-filled temp file used as the
child's stdin, so the full prompt is present the instant the child runs.

This test launches N stdin agents concurrently and asserts every child — the
first included — receives its exact, complete prompt with no cross-talk.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from spec_kitty_orchestrator.executor import (
    _close_prompt_file,
    execute_with_timeout,
    spawn_agent,
)


class _StdinToFileInvoker:
    """Fake stdin-based agent: copies whatever it gets on stdin into out_path."""

    agent_id = "stdin-echo"
    uses_stdin = True

    def __init__(self, out_path: Path) -> None:
        self._out = out_path

    def build_command(self, prompt: str, working_dir: Path, role: str) -> list[str]:
        return ["sh", "-c", f"cat > {shlex.quote(str(self._out))}"]


async def _run(invoker: _StdinToFileInvoker, prompt: str, wd: Path) -> None:
    process, _cmd, prompt_file = await spawn_agent(invoker, prompt, wd, "implementation")
    try:
        await execute_with_timeout(process, timeout_seconds=15)
    finally:
        _close_prompt_file(prompt_file)


def test_concurrent_stdin_agents_each_receive_full_prompt(tmp_path: Path) -> None:
    n = 6
    # Distinct, non-trivial prompts (>pipe-buffer-ish) so truncation/cross-talk shows.
    prompts = {i: f"PROMPT-{i}\n" + ("x" * 5000) + f"\nEND-{i}\n" for i in range(n)}
    outs = {i: tmp_path / f"out-{i}.txt" for i in range(n)}

    async def main() -> None:
        await asyncio.gather(
            *[_run(_StdinToFileInvoker(outs[i]), prompts[i], tmp_path) for i in range(n)]
        )

    asyncio.run(main())

    for i in range(n):
        assert outs[i].exists(), f"agent {i} produced no output (empty stdin?)"
        got = outs[i].read_text(encoding="utf-8")
        assert got == prompts[i], f"agent {i} did not receive its exact full prompt"
