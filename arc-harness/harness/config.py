"""Configuration for the ARC synthesis harness."""

import os
from dataclasses import dataclass, field


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file next to the project root.

    Existing environment variables take precedence. Enables keys to be
    shared across invocations without relying on the shell profile.
    """
    for candidate in (
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ):
        if not os.path.exists(candidate):
            continue
        with open(candidate) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


_load_dotenv()


@dataclass
class ModelConfig:
    """An OpenAI-compatible model endpoint."""

    name: str
    base_url: str
    api_key: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


def load_models() -> list[ModelConfig]:
    """Build the model registry from environment variables.

    Grok models route to the xAI API directly when XAI_API_KEY is set;
    otherwise they fall back to the RouteLLM (Abacus) endpoint.
    """
    base = os.environ.get("OPENAI_BASE_URL", "https://routellm.abacus.ai/v1")
    key = os.environ.get("OPENAI_API_KEY", "")
    xai_base = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    xai_key = os.environ.get("XAI_API_KEY", "")
    catalog = {
        "claude-opus-5": ModelConfig("claude-opus-5", base, key, 5, 25),
        "claude-sonnet-5": ModelConfig("claude-sonnet-5", base, key, 2, 10),
        "claude-sonnet-4-6": ModelConfig("claude-sonnet-4-6", base, key, 3, 15),
        "gpt-5.6-luna": ModelConfig("gpt-5.6-luna", base, key, 0.2, 1.2),
        "gpt-5.6-sol": ModelConfig("gpt-5.6-sol", base, key, 5, 30),
        "gpt-5.5": ModelConfig("gpt-5.5", base, key, 5, 30),
        "grok-4.6": ModelConfig("grok-4.6", xai_base if xai_key else base, xai_key or key, 2, 6),
        "grok-4.5": ModelConfig("grok-4.5", xai_base if xai_key else base, xai_key or key, 2, 6),
        "grok-4.3": ModelConfig("grok-4.3", xai_base if xai_key else base, xai_key or key, 1.25, 2.5),
        "gemini-3.6-flash": ModelConfig("gemini-3.6-flash", base, key, 1.5, 7.5),
        "deepseek-ai/DeepSeek-V4-Pro": ModelConfig("deepseek-ai/DeepSeek-V4-Pro", base, key, 1.74, 3.48),
    }
    return catalog


@dataclass
class HarnessConfig:
    """Top-level harness settings."""

    synth_model: str = "claude-opus-5"
    candidate_model: str = "gpt-5.6-luna"
    max_attempts: int = 8
    candidates_per_attempt: int = 2
    initial_candidates: int = 4
    max_tokens_generate: int = 4000
    temperature: float = 0.7
    refine_temperature: float = 0.5
    sandbox_timeout_s: float = 10.0
    llm_timeout_s: float = 600.0
    task_timeout_s: float = 1500.0
    reasoning_effort: str | None = None  # e.g. "low" for grok-4.6 fast path
    kfold: bool = False  # leave-one-out validation gate (diagnostic; off by default)
    fold_budget_attempts: int = 3  # per-fold refine budget
    fold_initial_candidates: int = 2  # per-fold first-round candidates
    max_folds: int = 4  # cap on number of leave-one-out folds
    consensus: bool = False  # consensus-on-test gate (diagnostic; off by default)
    consensus_rounds: int = 3  # independent full-train solves (incl. the main one)
    consensus_majority: int = 2  # min agreeing solvers required for acceptance
    consensus_budget_attempts: int = 2  # per-extra-round refine budget
    consensus_initial_candidates: int = 2  # per-extra-round first-round candidates
    consensus_round_timeout_s: float = 240.0  # per-extra-round wall-clock cap
    data_root: str = "/Users/evanpieser/arc-harness/data"
    library_root: str = "/Users/evanpieser/arc-harness/library"
    runs_root: str = "/Users/evanpieser/arc-harness/runs"
    use_library: bool = True
    top_k_library: int = 3
    seed: int = 0
    exclude_solved: bool = False
    num_workers: int = 4

    @property
    def run_dir(self) -> str:
        return self.runs_root

    # Where the verified solvers live (for library extraction + as exemplars)
    solvers_root: str = "/Users/evanpieser/arc-harness/data/arc_dataset"
