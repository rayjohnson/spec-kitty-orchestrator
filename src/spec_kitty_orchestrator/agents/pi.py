"""Pi coding agent invoker."""

from __future__ import annotations

from pathlib import Path

from .base import BaseInvoker, InvocationResult


class PiInvoker(BaseInvoker):
    """Invoker for Pi (`pi`).

    Pi print mode reads piped stdin and can emit JSON events as JSON lines.
    """

    agent_id = "pi"
    command = "pi"
    uses_stdin = True

    def build_command(self, prompt: str, working_dir: Path, role: str) -> list[str]:
        cmd = [
            "pi",
            "-p",
            "--mode", "json",
            "--no-session",
        ]
        if role == "review":
            cmd.extend(["--tools", "read,grep,find,ls"])
        elif role == "implementation":
            cmd.extend(["--tools", "read,bash,edit,write,grep,find,ls"])
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
