"""Shared LLM client for all step generators.

Supports anthropic and openai providers. Each step's generator.py
imports make_client() and call_llm() from here.
"""

from __future__ import annotations

SUPPORTED_PROVIDERS = ("anthropic", "openai")


def make_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    raise ValueError(f"Unknown provider: {provider}")


def call_llm(client, model: str, prompt: str, provider: str = "anthropic") -> str:
    if provider == "anthropic":
        response = client.messages.create(
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
        )
        blocks = [b.text for b in response.content if b.type == "text"]
        return "".join(blocks)
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


def check_api_key(provider: str) -> bool:
    import os
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY" in os.environ
    if provider == "openai":
        return "OPENAI_API_KEY" in os.environ
    return False
