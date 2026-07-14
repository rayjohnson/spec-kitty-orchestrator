"""Crash-safe single-process lock for orchestration runs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout as FileLockTimeout


class OrchestrationAlreadyRunningError(RuntimeError):
    """Raised when a live process owns the repository orchestration lock."""


def _read_owner(lock_file: Path) -> dict[str, object] | None:
    try:
        owner = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return owner if isinstance(owner, dict) else None


@contextmanager
def orchestration_lock(lock_file: Path, mission: str) -> Iterator[None]:
    """Hold a cross-platform kernel lock for one repository orchestration.

    ``filelock`` owns the crash-safe advisory lock; ``lock_file`` is diagnostic
    JSON only. A process crash releases the kernel lock automatically, so no
    stale-owner deletion or PID liveness race is needed.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    kernel_lock = FileLock(f"{lock_file}.guard")
    try:
        kernel_lock.acquire(timeout=0)
    except FileLockTimeout as exc:
        owner = _read_owner(lock_file) or {}
        raise OrchestrationAlreadyRunningError(
            f"orchestration already running for mission "
            f"'{owner.get('mission', 'unknown')}' (pid {owner.get('pid', 'unknown')})"
        ) from exc

    payload = json.dumps({"pid": os.getpid(), "mission": mission})
    lock_file.write_text(payload, encoding="utf-8")

    try:
        yield
    finally:
        lock_file.unlink(missing_ok=True)
        kernel_lock.release()


__all__ = ["OrchestrationAlreadyRunningError", "orchestration_lock"]
