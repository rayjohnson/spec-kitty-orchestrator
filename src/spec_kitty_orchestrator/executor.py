"""Subprocess spawning and log capture for agent executions.

Spawns agent processes asynchronously, captures stdout/stderr to log files,
and enforces timeouts. Uses workspace_path returned by the host API and creates
a provider-local git worktree when older hosts return a path without creating it.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
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


async def spawn_agent(
    invoker: BaseInvoker,
    prompt: str,
    working_dir: Path,
    role: str,
    repo_root: Path | None = None,
) -> tuple[asyncio.subprocess.Process, list[str]]:
    """Spawn an agent subprocess.

    Args:
        invoker: Agent invoker.
        prompt: Task prompt (sent via stdin if invoker.uses_stdin).
        working_dir: Directory where agent should run.
        role: "implementation" or "review".

    Returns:
        (process, cmd) tuple.

    Raises:
        ProcessSpawnError: If the process cannot be spawned.
    """
    ensure_working_dir(working_dir, repo_root)
    cmd = invoker.build_command(prompt, working_dir, role)
    logger.info("Spawning %s: %s ...", invoker.agent_id, " ".join(cmd[:3]))

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        logger.debug("Process %s spawned for %s", process.pid, invoker.agent_id)
        return process, cmd
    except OSError as exc:
        raise ProcessSpawnError(
            f"Failed to spawn {invoker.agent_id}: {exc}"
        ) from exc


async def execute_with_timeout(
    process: asyncio.subprocess.Process,
    stdin_data: bytes | None,
    timeout_seconds: int,
) -> tuple[bytes, bytes, int]:
    """Wait for process with timeout; kill gracefully if exceeded.

    Args:
        process: The spawned asyncio subprocess.
        stdin_data: Bytes to send to stdin (None if not uses_stdin).
        timeout_seconds: Maximum allowed execution time.

    Returns:
        (stdout_bytes, stderr_bytes, exit_code) — exit_code is TIMEOUT_EXIT_CODE
        if the process was killed due to timeout.
    """
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=stdin_data),
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
    stdin_data = prompt.encode("utf-8") if invoker.uses_stdin else None

    process, cmd = await spawn_agent(
        invoker, prompt, working_dir, role, repo_root=repo_root
    )
    stdout_bytes, stderr_bytes, exit_code = await execute_with_timeout(
        process, stdin_data, timeout_seconds
    )

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
    "TIMEOUT_EXIT_CODE",
]
