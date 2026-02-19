from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class OpenAIClient:
    api_key: str
    model: str
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or "No response returned by the model."


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: int = 120

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.2,
            },
        }

        response = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        response_json = response.json()

        message = response_json.get("message", {})
        return message.get("content", "No response returned by Ollama.")


def create_llm_client(
    provider: str,
    *,
    openai_api_key: str | None,
    openai_model: str,
    ollama_base_url: str,
    ollama_model: str,
) -> LLMClient:
    provider_normalized = provider.strip().lower()

    if provider_normalized == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when provider is OpenAI.")
        return OpenAIClient(api_key=openai_api_key, model=openai_model)

    if provider_normalized == "ollama":
        return OllamaClient(base_url=ollama_base_url, model=ollama_model)

    raise ValueError(f"Unsupported provider: {provider}")

