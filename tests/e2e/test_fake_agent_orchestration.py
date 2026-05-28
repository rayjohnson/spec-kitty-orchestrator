from __future__ import annotations

import pytest

from .harness import final_wp_lane, read_status_events, require_success, run_orchestrator


pytestmark = pytest.mark.e2e


def assert_done_with_policy(project) -> None:
    assert final_wp_lane(project.mission_dir) == "done"
    events = read_status_events(project.mission_dir)
    assert any(event.get("to_lane") == "in_progress" for event in events)
    assert any(event.get("to_lane") == "for_review" for event in events)
    assert any(event.get("to_lane") == "done" for event in events)
    assert any(event.get("policy_metadata") for event in events), events
    assert (project.root / ".kittify" / "orchestrator-run-state.json").exists()
    assert list((project.root / ".kittify" / "logs").glob("*.log"))


@pytest.mark.parametrize(
    ("impl_agent", "review_agent"),
    [
        ("claude-code", "codex"),
        ("claude-code", "opencode"),
        ("codex", "claude-code"),
        ("opencode", "claude-code"),
    ],
)
def test_orchestrate_happy_path_with_fake_primary_agents(fake_agent_project, impl_agent: str, review_agent: str) -> None:
    result = run_orchestrator(fake_agent_project, impl_agent=impl_agent, review_agent=review_agent)

    require_success(result)
    assert_done_with_policy(fake_agent_project)


def test_orchestrate_rejection_rework_cycle_with_fake_agents(fake_agent_project) -> None:
    result = run_orchestrator(
        fake_agent_project,
        impl_agent="claude-code",
        review_agent="codex",
        extra_env={"SK_ORCH_FAKE_REVIEW_FAIL_ONCE": "1"},
    )

    require_success(result)
    assert_done_with_policy(fake_agent_project)
    events = read_status_events(fake_agent_project.mission_dir)
    assert any(event.get("review_ref") for event in events), events


def test_orchestrate_writes_status_for_blocked_implementation_failure(fake_agent_project) -> None:
    # Replace fake Claude with a deterministic failing implementation agent.
    failing = fake_agent_project.bin_dir / "claude"
    failing.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "print(json.dumps({'error': 'implementation failed for e2e'}))\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    failing.chmod(0o755)

    result = run_orchestrator(fake_agent_project, impl_agent="claude-code", review_agent="codex")

    assert result.returncode != 0 or final_wp_lane(fake_agent_project.mission_dir) == "blocked"
    assert final_wp_lane(fake_agent_project.mission_dir) == "blocked"


def test_status_resume_and_abort_commands_use_provider_state(fake_agent_project) -> None:
    result = run_orchestrator(fake_agent_project, impl_agent="claude-code", review_agent="codex")
    require_success(result)

    status = run_orchestrator_status(fake_agent_project)
    require_success(status)
    assert "WP01" in status.stdout

    resume = run_orchestrator_resume(fake_agent_project)
    require_success(resume)

    abort = run_orchestrator_abort(fake_agent_project)
    require_success(abort)
    assert not (fake_agent_project.root / ".kittify" / "orchestrator-run-state.json").exists()


def run_orchestrator_status(project):
    import sys
    from .harness import run_command

    return run_command(
        [
            sys.executable,
            "-m",
            "spec_kitty_orchestrator.cli.main",
            "status",
            "--repo-root",
            str(project.root),
        ],
        cwd=project.root,
        env=project.env,
    )


def run_orchestrator_resume(project):
    import sys
    from .harness import run_command

    return run_command(
        [
            sys.executable,
            "-m",
            "spec_kitty_orchestrator.cli.main",
            "resume",
            "--repo-root",
            str(project.root),
        ],
        cwd=project.root,
        env=project.env,
    )


def run_orchestrator_abort(project):
    import sys
    from .harness import run_command

    return run_command(
        [
            sys.executable,
            "-m",
            "spec_kitty_orchestrator.cli.main",
            "abort",
            "--cleanup-worktrees",
            "--repo-root",
            str(project.root),
        ],
        cwd=project.root,
        env=project.env,
    )
