"""Configuration for the Computer Agent framework."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AgentConfig:
    """
    Central configuration for the computer agent.

    Attributes:
        api_key:        LLM API key (or set via env vars)
        provider:       LLM provider ('openai', 'anthropic', 'mercury')
        model:          Model name override
        base_url:       API base URL override
        screenshot_dir: Directory for saving screenshots
        max_steps:      Maximum agent loop iterations (safety bound)
        step_delay:     Seconds to wait between actions
        confirm_actions: Require user confirmation before executing
        safe_mode:      Restrict to bounded region only
        bounds:         (x, y, width, height) restriction region
        screenshot_scale: Downscale factor for screenshots (1.0 = full res)
        verbose:        Print detailed action logs
        temperature:    LLM temperature
        max_tokens:     Max LLM response tokens
    """
    api_key: Optional[str] = None
    provider: str = "openai"
    model: Optional[str] = None
    base_url: Optional[str] = None
    screenshot_dir: str = "/tmp/computer_agent_screenshots"
    max_steps: int = 50
    step_delay: float = 1.0
    confirm_actions: bool = False
    safe_mode: bool = False
    bounds: Optional[Tuple[int, int, int, int]] = None
    screenshot_scale: float = 1.0
    verbose: bool = True
    temperature: float = 0.0
    max_tokens: int = 4096

    # Provider defaults
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "vision": True,
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "default_model": "claude-sonnet-4-20250514",
            "vision": True,
        },
        "mercury": {
            "base_url": "https://api.inceptionlabs.ai/v1",
            "default_model": "mercury-coder-small",
            "vision": False,  # Mercury 2 is text-only for now
        },
    }

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return self.PROVIDERS.get(self.provider, self.PROVIDERS["openai"])["base_url"]

    def get_model(self) -> str:
        if self.model:
            return self.model
        return self.PROVIDERS.get(self.provider, self.PROVIDERS["openai"])["default_model"]

    def supports_vision(self) -> bool:
        return self.PROVIDERS.get(self.provider, {}).get("vision", False)
