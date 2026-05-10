"""Shared LLM client for all step generators.

Supports anthropic and openai providers. Each step's generator.py
imports make_client() and call_llm() from here.
"""

from __future__ import annotations

SUPPORTED_PROVIDERS = ("anthropic", "openai")
SUBSCRIPTION_BACKENDS = ("claude", "codex")
API_BACKENDS = ("anthropic-api", "openai-api")
SUPPORTED_BACKENDS = SUBSCRIPTION_BACKENDS + API_BACKENDS


def make_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    raise ValueError(f"Unknown provider: {provider}")


def call_llm(
    client, model: str, prompt: str, provider: str = "anthropic",
    max_retries: int = 3,
) -> str:
    import time

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "anthropic":
                with client.messages.stream(
                    model=model,
                    max_tokens=64_000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                ) as stream:
                    return stream.get_final_text()
            if provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            raise ValueError(f"Unknown provider: {provider}")
        except (ConnectionError, OSError) as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"[llm] connection error (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
        except Exception as e:
            if "ReadError" in type(e).__name__ or "10054" in str(e):
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                print(f"[llm] network drop (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def check_api_key(provider: str) -> bool:
    import os
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY" in os.environ
    if provider == "openai":
        return "OPENAI_API_KEY" in os.environ
    return False


def default_model_for_backend(backend: str, role: str = "generator") -> str:
    if backend in {"claude", "anthropic-api"}:
        if role in {"verifier", "judge"}:
            return "claude-sonnet-4-6"
        return "claude-opus-4-7"
    if backend in {"codex", "openai-api"}:
        if role == "evaluator":
            return "gpt-5-mini"
        if role == "judge":
            return "gpt-5.4"
        if role == "verifier":
            return "gpt-5"
        return "gpt-5.5"
    raise ValueError(f"Unsupported backend: {backend}")


def provider_for_backend(backend: str, provider: str = "anthropic") -> str:
    if backend == "anthropic-api":
        return "anthropic"
    if backend == "openai-api":
        return "openai"
    if backend == "api":
        return provider
    raise ValueError(f"Backend {backend!r} is not an API backend")


def call_claude_cli(
    prompt: str,
    model: str | None = None,
    claude_cmd: str | None = None,
    max_retries: int = 2,
) -> str:
    """Call `claude -p` as a pure text subprocess."""
    import os
    import subprocess
    import sys
    import tempfile

    cmd_name = claude_cmd or os.environ.get(
        "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
    )
    cmd = [cmd_name, "-p", "--tools", ""]
    if model:
        cmd += ["--model", model]

    last_stderr = ""
    for attempt in range(max_retries + 1):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(prompt)
        tmp.close()
        try:
            with open(tmp.name, "r", encoding="utf-8") as stream:
                result = subprocess.run(
                    cmd,
                    stdin=stream,
                    capture_output=True,
                    timeout=1800,
                )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            if result.returncode == 0 and stdout.strip():
                return stdout.strip()
            last_stderr = stderr
            if attempt < max_retries:
                print(
                    f"  retry {attempt + 1}/{max_retries} "
                    f"(stderr: {stderr[:200]})"
                )
        finally:
            os.unlink(tmp.name)
    raise RuntimeError(
        f"claude CLI failed after {max_retries + 1} attempts: {last_stderr[:500]}"
    )


def call_codex_cli(
    prompt: str,
    model: str | None = None,
    codex_cmd: str | None = None,
    max_retries: int = 2,
) -> str:
    """Call `codex exec` as an isolated subscription-backed subprocess."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    cmd_name = codex_cmd or os.environ.get(
        "CODEX_CMD", "codex.exe" if sys.platform == "win32" else "codex"
    )
    wrapped_prompt = (
        "You are being used as a pure inference backend for PersonaBench. "
        "Do not inspect files, run commands, browse, or use tools. Read only "
        "the prompt below and return only the requested JSON.\n\n"
        + prompt
    )

    last_stderr = ""
    for attempt in range(max_retries + 1):
        with tempfile.TemporaryDirectory(prefix="personabench_codex_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_path = tmp_path / "last_message.txt"
            cmd = [
                cmd_name,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "-C",
                str(tmp_path),
                "-o",
                str(out_path),
            ]
            if model:
                cmd += ["--model", model]
            cmd.append("-")
            result = subprocess.run(
                cmd,
                input=wrapped_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=1800,
            )
            if result.returncode == 0:
                if out_path.exists():
                    text = out_path.read_text(encoding="utf-8").strip()
                    if text:
                        return text
                if result.stdout.strip():
                    return result.stdout.strip()
            last_stderr = result.stderr
            if attempt < max_retries:
                print(
                    f"  retry {attempt + 1}/{max_retries} "
                    f"(stderr: {last_stderr[:200]})"
                )
    raise RuntimeError(
        f"codex CLI failed after {max_retries + 1} attempts: {last_stderr[:500]}"
    )


def call_subscription_cli(
    prompt: str,
    model: str | None,
    backend: str,
    claude_cmd: str | None = None,
    codex_cmd: str | None = None,
    max_retries: int = 2,
) -> str:
    if backend == "claude":
        return call_claude_cli(prompt, model, claude_cmd, max_retries=max_retries)
    if backend == "codex":
        return call_codex_cli(prompt, model, codex_cmd, max_retries=max_retries)
    raise ValueError(f"Unknown subscription backend: {backend}")
