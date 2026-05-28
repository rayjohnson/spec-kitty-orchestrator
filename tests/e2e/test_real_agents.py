from __future__ import annotations

import os

import pytest

from .harness import final_wp_lane, has_real_agent, require_success, run_orchestrator, truthy


pytestmark = [pytest.mark.e2e, pytest.mark.real_agents, pytest.mark.trusted_runner, pytest.mark.slow]


DEFAULT_MATRIX = [
    ("claude-code", "codex"),
    ("claude-code", "opencode"),
    ("codex", "claude-code"),
    ("opencode", "claude-code"),
]


def selected_matrix() -> list[tuple[str, str]]:
    raw = os.environ.get("SK_ORCH_E2E_AGENT_MATRIX")
    if not raw:
        return DEFAULT_MATRIX
    pairs: list[tuple[str, str]] = []
    for chunk in raw.split(","):
        left, sep, right = chunk.partition(":")
        if sep:
            pairs.append((left.strip(), right.strip()))
    return pairs or DEFAULT_MATRIX


@pytest.mark.parametrize(("impl_agent", "review_agent"), selected_matrix())
def test_real_agent_matrix(real_agent_project, impl_agent: str, review_agent: str) -> None:
    if not truthy(os.environ.get("SK_ORCH_E2E_REAL_AGENTS")):
        pytest.skip("Set SK_ORCH_E2E_REAL_AGENTS=1 to run real-agent orchestrator e2e tests.")
    for agent in {impl_agent, review_agent}:
        if not has_real_agent(agent):
            pytest.skip(f"Agent CLI for {agent!r} is not installed on PATH.")

    timeout = int(os.environ.get("SK_ORCH_E2E_TIMEOUT_SECONDS", "3600"))
    result = run_orchestrator(
        real_agent_project,
        impl_agent=impl_agent,
        review_agent=review_agent,
        timeout=timeout,
    )

    require_success(result)
    assert final_wp_lane(real_agent_project.mission_dir) == "done"
