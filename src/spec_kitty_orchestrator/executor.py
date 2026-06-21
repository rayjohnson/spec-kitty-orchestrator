"""Subprocess spawning and log capture for agent executions.

Spawns agent processes asynchronously, captures stdout/stderr to log files,
and enforces timeouts. Uses workspace_path returned by the host API and creates
a provider-local git worktree when older hosts return a path without creating it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from .agents.base import AgentInvoker, BaseInvoker, InvocationResult

logger = logging.getLogger(__name__)

TIMEOUT_EXIT_CODE = 124
TERMINATION_GRACE_SECONDS = 5.0


class ExecutorError(Exception):
    """Base exception for executor errors."""


class ProcessSpawnError(ExecutorError):
    """Raised when process spawning fails."""


class ExecutionTimeoutError(ExecutorError):
    """Raised when an agent execution exceeds the timeout."""


class CommitError(ExecutorError):
    """Raised when committing the lane worktree fails or is unsafe."""


def _git(workspace_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def current_lane_head(workspace_path: Path) -> str | None:
    """Return the lane worktree's current HEAD sha, or None if it can't be read."""
    head = _git(workspace_path, ["rev-parse", "HEAD"])
    return head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else None


def commit_lane_work(
    workspace_path: Path,
    message: str,
    *,
    lane_branch: str | None,
    output_base: str | None,
) -> bool:
    """Stage and commit the agent's edits in the lane worktree.

    Returns True when this work package produced real output — i.e. there is at
    least one commit between ``output_base`` and HEAD after committing. Returns
    False when nothing was produced (an empty/no-op WP that must not advance).

    Args:
        workspace_path: The lane worktree (must be checked out on ``lane_branch``).
        message: Commit message.
        lane_branch: Expected branch the worktree is on. When set, the worktree is
            verified to be on it first — guarding against committing to a stray
            detached/branchless worktree.
        output_base: The lane HEAD captured *before* this WP's implementation ran
            (so dependency-lane merges already applied at allocation are NOT
            counted as this WP's output). Output is measured as
            ``output_base..HEAD``. When None, falls back to "did this call stage
            and commit anything".

    Raises:
        CommitError: if the worktree is not on ``lane_branch`` or git fails.
    """
    if lane_branch:
        head = _git(workspace_path, ["rev-parse", "--abbrev-ref", "HEAD"])
        current = head.stdout.strip()
        if head.returncode != 0 or current != lane_branch:
            raise CommitError(
                f"Lane worktree {workspace_path} is on '{current or 'DETACHED'}', "
                f"expected lane branch '{lane_branch}'. Refusing to commit — the "
                "host must allocate the lane worktree on its lane branch."
            )

    add = _git(workspace_path, ["add", "-A"])
    if add.returncode != 0:
        raise CommitError(f"git add failed in {workspace_path}: {add.stderr.strip()}")

    has_staged = _git(workspace_path, ["diff", "--cached", "--quiet"]).returncode != 0
    if has_staged:
        commit = _git(workspace_path, ["commit", "-m", message])
        if commit.returncode != 0:
            raise CommitError(
                f"git commit failed in {workspace_path}: "
                f"{commit.stderr.strip() or commit.stdout.strip()}"
            )

    if output_base:
        count = _git(workspace_path, ["rev-list", "--count", f"{output_base}..HEAD"])
        if count.returncode != 0:
            raise CommitError(
                f"git rev-list failed in {workspace_path}: {count.stderr.strip()}"
            )
        try:
            return int(count.stdout.strip()) > 0
        except ValueError:
            return has_staged
    return has_staged


def ensure_working_dir(working_dir: Path, repo_root: Path | None = None) -> None:
    """Ensure the subprocess cwd exists and is usable.

    Some host contract versions return a .worktrees path from start-implementation
    without creating it. Prefer a detached git worktree so agents can commit; fall
    back to a plain directory only when no git repo root is available.
    """
    if working_dir.exists():
        if not working_dir.is_dir():
            raise ProcessSpawnError(
                f"Working directory is not a directory: {working_dir}"
            )
        return

    if repo_root is not None and _is_git_repo(repo_root):
        working_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--detach",
                str(working_dir),
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Created missing agent worktree at %s", working_dir)
            return
        raise ProcessSpawnError(
            "Failed to create working directory as git worktree "
            f"{working_dir}: {result.stderr.strip() or result.stdout.strip()}"
        )

    try:
        working_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProcessSpawnError(
            f"Failed to create working directory {working_dir}: {exc}"
        ) from exc


def _is_git_repo(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def get_log_path(log_dir: Path, mission: str, wp_id: str, role: str) -> Path:
    """Return the log file path for a given WP execution.

    Args:
        log_dir: Base log directory (provider-owned).
        mission: Mission slug.
        wp_id: Work package ID.
        role: "implementation" or "review".

    Returns:
        Path to the log file (not yet created).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{mission}_{wp_id}_{role}.log"


def _close_prompt_file(prompt_file: "tempfile._TemporaryFileWrapper | None") -> None:
    """Close and remove a temp prompt file (best-effort)."""
    if prompt_file is None:
        return
    name = prompt_file.name
    with suppress(OSError):
        prompt_file.close()
    with suppress(OSError):
        os.unlink(name)


async def spawn_agent(
    invoker: BaseInvoker,
    prompt: str,
    working_dir: Path,
    role: str,
    repo_root: Path | None = None,
) -> tuple[asyncio.subprocess.Process, list[str], "tempfile._TemporaryFileWrapper | None"]:
    """Spawn an agent subprocess.

    For stdin-based invokers the prompt is delivered via a pre-filled temp file
    used as the child's stdin, NOT by writing a PIPE after the process starts.
    Writing a PIPE post-spawn races under concurrent launches: all the children
    start before any write happens, so the first-spawned ``claude -p`` reads an
    empty/closed stdin and aborts ("Input must be provided ... when using
    --print"). A file fd is complete and readable the instant the child runs.

    Returns:
        (process, cmd, prompt_file) — ``prompt_file`` is the temp file backing
        stdin (or None); the caller must close it via ``_close_prompt_file``
        once the process has finished.

    Raises:
        ProcessSpawnError: If the process cannot be spawned.
    """
    ensure_working_dir(working_dir, repo_root)
    cmd = invoker.build_command(prompt, working_dir, role)
    logger.info("Spawning %s: %s ...", invoker.agent_id, " ".join(cmd[:3]))

    prompt_file: "tempfile._TemporaryFileWrapper | None" = None
    if invoker.uses_stdin:
        prompt_file = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", suffix=".prompt", delete=False
        )
        prompt_file.write(prompt)
        prompt_file.flush()
        prompt_file.seek(0)
        stdin_arg: int = prompt_file.fileno()
    else:
        stdin_arg = asyncio.subprocess.DEVNULL

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=stdin_arg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
    except OSError as exc:
        _close_prompt_file(prompt_file)
        raise ProcessSpawnError(
            f"Failed to spawn {invoker.agent_id}: {exc}"
        ) from exc
    logger.debug("Process %s spawned for %s", process.pid, invoker.agent_id)
    return process, cmd, prompt_file


async def execute_with_timeout(
    process: asyncio.subprocess.Process,
    timeout_seconds: int,
) -> tuple[bytes, bytes, int]:
    """Wait for process with timeout; kill gracefully if exceeded.

    Stdin is delivered at spawn time (via a file for stdin-based invokers, or
    DEVNULL otherwise), so this only drains stdout/stderr and waits.

    Returns:
        (stdout_bytes, stderr_bytes, exit_code) — exit_code is TIMEOUT_EXIT_CODE
        if the process was killed due to timeout.
    """
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=float(timeout_seconds),
        )
        return stdout_bytes, stderr_bytes, process.returncode or 0
    except asyncio.TimeoutError:
        logger.warning("Process %s timed out after %ss", process.pid, timeout_seconds)
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE_SECONDS)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return b"", b"", TIMEOUT_EXIT_CODE


async def execute_agent(
    invoker: BaseInvoker,
    prompt: str,
    working_dir: Path,
    role: str,
    timeout_seconds: int,
    log_file: Path | None = None,
    repo_root: Path | None = None,
) -> InvocationResult:
    """Execute an agent and return a structured InvocationResult.

    Handles stdin piping, timeout, log capture.

    Args:
        invoker: Agent invoker instance.
        prompt: Task prompt text.
        working_dir: Directory for agent execution.
        role: "implementation" or "review".
        timeout_seconds: Maximum execution time.
        log_file: Optional path to write combined stdout+stderr.
        repo_root: Optional repository root used to create a missing worktree cwd.

    Returns:
        InvocationResult with all captured output.
    """
    start = time.monotonic()

    process, cmd, prompt_file = await spawn_agent(
        invoker, prompt, working_dir, role, repo_root=repo_root
    )
    try:
        stdout_bytes, stderr_bytes, exit_code = await execute_with_timeout(
            process, timeout_seconds
        )
    finally:
        _close_prompt_file(prompt_file)

    duration = time.monotonic() - start
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as fh:
                fh.write(f"=== command: {' '.join(cmd)} ===\n")
                fh.write(f"=== exit_code: {exit_code} ===\n")
                fh.write(f"=== stdout ===\n{stdout}\n")
                fh.write(f"=== stderr ===\n{stderr}\n")
        except OSError as exc:
            logger.warning("Failed to write log file %s: %s", log_file, exc)

    logger.info(
        "%s %s/%s finished: exit=%d, duration=%.1fs",
        invoker.agent_id, role, working_dir.name, exit_code, duration,
    )
    return invoker.parse_output(stdout, stderr, exit_code, duration)


__all__ = [
    "ExecutorError",
    "ProcessSpawnError",
    "ExecutionTimeoutError",
    "ensure_working_dir",
    "get_log_path",
    "execute_agent",
    "commit_lane_work",
    "current_lane_head",
    "CommitError",
    "TIMEOUT_EXIT_CODE",
]
