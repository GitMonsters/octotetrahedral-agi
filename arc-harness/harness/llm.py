"""Thin OpenAI-compatible client wrapper with token accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .config import ModelConfig


@dataclass
class LLMUsage:
    """Accumulated token usage and cost for a model."""

    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, cfg: ModelConfig) -> float:
        return (
            self.input_tokens / 1e6 * cfg.input_per_mtok
            + self.output_tokens / 1e6 * cfg.output_per_mtok
        )

    def add(self, usage) -> None:
        if not usage:
            return
        in_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        out_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        self.input_tokens += int(in_tokens)
        self.output_tokens += int(out_tokens)


@dataclass
class LLMResult:
    text: str
    usage: Optional[LLMUsage] = None


class LLMClient:
    """Minimal OpenAI-compatible client."""

    def __init__(self, cfg: ModelConfig, timeout: float = 600.0):
        self.cfg = cfg
        self.timeout = timeout
        # reasoning models (grok-4.6, etc.) can take minutes of hidden
        # reasoning before the first visible token, so keep a generous timeout
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=timeout)
        self.usage = LLMUsage()

    def _estimate(self, text: str) -> int:
        """Rough token estimate (chars/4) when the router omits usage."""
        return max(1, len(text) // 4)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMResult:
        model = model or self.cfg.name
        if timeout is not None and timeout < self.timeout:
            self.client.timeout = timeout
        extra = {}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra or None,
        )
        if resp.usage is None or (getattr(resp.usage, "prompt_tokens", None) is None
                                  and getattr(resp.usage, "input_tokens", None) is None):
            in_tokens = sum(self._estimate(m.get("content") or "") for m in messages)
            out_text = resp.choices[0].message.content or ""
            out_tokens = self._estimate(out_text)
            self.usage.input_tokens += in_tokens
            self.usage.output_tokens += out_tokens
        else:
            self.usage.add(resp.usage)
        return LLMResult(
            text=(resp.choices[0].message.content or ""),
            usage=self.usage,
        )

    def cost_so_far(self) -> float:
        return self.usage.cost(self.cfg)
