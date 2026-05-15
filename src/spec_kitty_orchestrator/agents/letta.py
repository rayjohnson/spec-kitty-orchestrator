"""Letta Code agent invoker."""

from __future__ import annotations

from pathlib import Path

from .base import BaseInvoker, InvocationResult


class LettaInvoker(BaseInvoker):
    """Invoker for Letta Code (`letta`).

    Letta headless mode reads piped stdin with ``-p`` and supports structured
    JSON output.
    """

    agent_id = "letta"
    command = "letta"
    uses_stdin = True

    def build_command(self, prompt: str, working_dir: Path, role: str) -> list[str]:
        cmd = [
            "letta",
            "-p",
            "--output-format", "json",
        ]
        if role == "implementation":
            cmd.append("--yolo")
        elif role == "review":
            cmd.extend(["--permission-mode", "plan", "--tools", "Read,Glob,Grep"])
        return cmd

    def parse_output(
        self, stdout: str, stderr: str, exit_code: int, duration_seconds: float
    ) -> InvocationResult:
        success = exit_code == 0
        data = self._parse_json_output(stdout)
        return InvocationResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            files_modified=self._extract_files_from_output(data),
            commits_made=self._extract_commits_from_output(data),
            errors=self._extract_errors_from_output(data, stderr),
            warnings=self._extract_warnings_from_output(data, stderr),
        )
