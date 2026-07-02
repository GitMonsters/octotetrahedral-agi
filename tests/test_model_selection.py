"""Integration tests for the model selection and integration layer.

Covers:
  1.  CLI parameter parsing (--model unified-stack)
  2.  Model registry — load, discover, fallback
  3.  Config file loading (YAML parsing, defaults)
  4.  Adapter — request conversion, output preservation
  5.  Chat store — state management, persistence (via modelStore logic)
  6.  Model switching mid-task
  7.  Coherence tracking with different models
  8.  Error handling (unavailable model → fallback)
  9.  Performance consistency
  10. Capability matching (task domain → best model)
  11. Version syntax (unified-stack:v1.2, unified-stack:16-limb)
  12. Model listing / discovery
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """Return a new ModelRegistry instance (avoids shared singleton state)."""
    from model_registry import ModelRegistry
    return ModelRegistry()


# ===========================================================================
# 1. CLI parameter parsing
# ===========================================================================

class TestCLIParameterParsing:
    def test_model_flag_sets_canonical_name(self):
        from cli_model_selector import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "unified-stack", "--task", "hello"])
        assert args.model == "unified-stack"

    def test_model_flag_accepts_variant_syntax(self):
        from cli_model_selector import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "unified-stack:16-limb", "--task", "test"])
        assert args.model == "unified-stack:16-limb"

    def test_list_models_flag(self):
        from cli_model_selector import build_parser
        parser = build_parser()
        args = parser.parse_args(["--list-models"])
        assert args.list_models is True

    def test_json_output_flag(self):
        from cli_model_selector import build_parser
        parser = build_parser()
        args = parser.parse_args(["--list-models", "--json-output"])
        assert args.json_output is True

    def test_no_persist_flag(self):
        from cli_model_selector import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "gpt-4", "--no-persist"])
        assert args.persist is False

    def test_main_list_models_returns_zero(self, capsys):
        from cli_model_selector import main
        rc = main(["--list-models"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "unified-stack" in captured.out

    def test_main_unknown_model_falls_back_and_succeeds(self, capsys):
        """An unknown model spec should trigger fallback, not a failure."""
        from cli_model_selector import main
        rc = main(["--model", "does-not-exist", "--task", "hi"])
        # Fallback is the expected behaviour; the run should succeed (rc == 0)
        assert rc == 0
        captured = capsys.readouterr()
        # The output should show the fallback model was used
        assert "unified-stack" in captured.out or "gpt-4" in captured.out


# ===========================================================================
# 2. Model registry — load, discover, fallback
# ===========================================================================

class TestModelRegistry:
    def test_default_models_registered(self):
        reg = _fresh_registry()
        names = {m.name for m in reg.list_models()}
        assert "unified-stack" in names
        assert "unified-stack-16" in names
        assert "gpt-4" in names
        assert "claude-3-opus" in names

    def test_load_unified_stack_returns_forward_model(self):
        reg = _fresh_registry()
        model = reg.load("unified-stack")
        assert model is not None
        assert hasattr(model, "forward")

    def test_load_external_model_returns_none(self):
        reg = _fresh_registry()
        result = reg.load("gpt-4")
        assert result is None

    def test_get_metadata_returns_correct_limb_count(self):
        reg = _fresh_registry()
        meta = reg.get_metadata("unified-stack")
        assert meta.limbs == 8

        meta16 = reg.get_metadata("unified-stack-16")
        assert meta16.limbs == 16

    def test_fallback_to_default_when_model_missing(self):
        reg = _fresh_registry()
        canonical = reg.with_fallback("nonexistent-model")
        assert canonical == "unified-stack"

    def test_fallback_chain_order(self):
        reg = _fresh_registry()
        reg.set_fallback_chain(["gpt-4", "claude-3-opus"])
        canonical = reg.with_fallback("nonexistent-model")
        assert canonical == "gpt-4"

    def test_custom_model_registration(self):
        from model_registry import ModelMetadata
        reg = _fresh_registry()
        reg.register(
            ModelMetadata(name="my-model", description="custom", capabilities=["reasoning"])
        )
        assert reg.is_available("my-model")

    def test_is_available_returns_false_for_unknown(self):
        reg = _fresh_registry()
        assert reg.is_available("definitely-not-a-real-model") is False


# ===========================================================================
# 3. Config file loading
# ===========================================================================

class TestConfigFileLoading:
    def test_load_config_returns_empty_dict_when_file_missing(self):
        from integration.copilot_integration import load_config
        result = load_config(Path("/tmp/does_not_exist_abc123.yml"))
        assert result == {}

    def test_load_config_parses_yaml(self):
        pytest.importorskip("yaml")
        from integration.copilot_integration import load_config
        content = "default_model: unified-stack\nfallback_chain:\n  - unified-stack\n  - gpt-4\n"
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(content)
            tmp = Path(f.name)
        try:
            cfg = load_config(tmp)
            assert cfg["default_model"] == "unified-stack"
            assert cfg["fallback_chain"] == ["unified-stack", "gpt-4"]
        finally:
            tmp.unlink(missing_ok=True)

    def test_bootstrap_uses_default_model_from_config(self):
        from integration.copilot_integration import bootstrap
        cfg = {"default_model": "gpt-4", "models": {}}
        integration = bootstrap(config=cfg)
        assert integration._adapter._default_model == "gpt-4"

    def test_bootstrap_registers_config_models(self):
        from integration.copilot_integration import bootstrap
        cfg = {
            "default_model": "gpt-4",
            "models": {
                "custom-test-model": {
                    "description": "test",
                    "limbs": 0,
                    "capabilities": ["reasoning"],
                }
            },
        }
        integration = bootstrap(config=cfg)
        assert integration.registry.is_available("custom-test-model")


# ===========================================================================
# 4. Adapter — request conversion, output preservation
# ===========================================================================

class TestCopilotModelAdapter:
    def _make_adapter(self, registry=None):
        from integration.copilot_model_adapter import CopilotModelAdapter
        return CopilotModelAdapter(registry=registry or _fresh_registry(), default_model="unified-stack")

    def test_to_unified_format_returns_correct_limb_count(self):
        adapter = self._make_adapter()
        req = {"prompt": "test prompt", "context": {}}
        result = adapter.to_unified_format(req, limb_count=8)
        assert len(result["limb_states"]) == 8
        assert result["task_signal"] == "test prompt"

    def test_to_unified_format_16_limbs(self):
        adapter = self._make_adapter()
        req = {"prompt": "test", "context": {}}
        result = adapter.to_unified_format(req, limb_count=16)
        assert len(result["limb_states"]) == 16

    def test_process_unified_stack_returns_coherence(self):
        adapter = self._make_adapter()
        req = {"prompt": "explain recursion", "model": "unified-stack"}
        response = adapter.process(req)
        assert "coherence" in response
        assert 0.0 <= response["coherence"] <= 1.0

    def test_process_external_model_returns_response(self):
        adapter = self._make_adapter()
        req = {"prompt": "hello", "model": "gpt-4"}
        response = adapter.process(req)
        assert response["model"] == "gpt-4"
        assert "latency_ms" in response

    def test_process_preserves_trace_id(self):
        adapter = self._make_adapter()
        req = {"prompt": "trace test", "model": "unified-stack", "trace_id": "abc-123"}
        response = adapter.process(req)
        assert response["trace_id"] == "abc-123"

    def test_process_unknown_model_triggers_fallback(self):
        adapter = self._make_adapter()
        req = {"prompt": "test", "model": "nonexistent-xyz"}
        response = adapter.process(req)
        # Should fall back to unified-stack (first in default chain)
        assert response.get("error") is None or response["model"] in {
            "unified-stack", "gpt-4", "claude-3-opus"
        }

    def test_limb_metadata_populated_for_unified_stack(self):
        adapter = self._make_adapter()
        req = {"prompt": "limb test", "model": "unified-stack"}
        response = adapter.process(req)
        lm = response.get("limb_metadata", {})
        assert "limb_states" in lm
        assert len(lm["limb_states"]) == 8


# ===========================================================================
# 5. Model switching mid-task
# ===========================================================================

class TestModelSwitchingMidTask:
    def test_adapter_uses_per_request_model_override(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        reg = _fresh_registry()
        adapter = CopilotModelAdapter(registry=reg, default_model="gpt-4")

        r1 = adapter.process({"prompt": "task 1", "model": "gpt-4"})
        r2 = adapter.process({"prompt": "task 2", "model": "unified-stack"})

        assert r1["model"] == "gpt-4"
        assert r2["model"] == "unified-stack"

    def test_integration_process_accepts_model_override(self):
        from integration.copilot_integration import bootstrap
        integration = bootstrap(config={"default_model": "gpt-4", "models": {}})

        r1 = integration.process_request({"prompt": "gpt request"})
        r2 = integration.process_request({"prompt": "stack request", "model": "unified-stack"})

        assert r1["model"] == "gpt-4"
        assert r2["model"] == "unified-stack"


# ===========================================================================
# 6. Coherence tracking with different models
# ===========================================================================

class TestCoherenceTracking:
    def test_unified_stack_coherence_in_valid_range(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        adapter = CopilotModelAdapter(registry=_fresh_registry(), default_model="unified-stack")
        response = adapter.process({"prompt": "coherence test", "model": "unified-stack"})
        assert 0.0 <= response["coherence"] <= 1.0

    def test_unified_stack_16_coherence_in_valid_range(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        adapter = CopilotModelAdapter(registry=_fresh_registry(), default_model="unified-stack-16")
        response = adapter.process({"prompt": "16-limb coherence", "model": "unified-stack-16"})
        assert 0.0 <= response["coherence"] <= 1.0

    def test_external_model_coherence_is_zero(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        adapter = CopilotModelAdapter(registry=_fresh_registry(), default_model="gpt-4")
        response = adapter.process({"prompt": "external test", "model": "gpt-4"})
        assert response["coherence"] == 0.0


# ===========================================================================
# 7. Error handling — unavailable model → fallback
# ===========================================================================

class TestErrorHandling:
    def test_fallback_on_unknown_model(self):
        reg = _fresh_registry()
        canonical = reg.with_fallback("nonexistent-abc")
        assert canonical in {"unified-stack", "gpt-4", "claude-3-opus"}

    def test_runtime_error_when_chain_exhausted(self):
        reg = _fresh_registry()
        reg.set_fallback_chain([])
        # Clear all registered models
        reg._metadata.clear()
        with pytest.raises(RuntimeError):
            reg.with_fallback("anything")

    def test_adapter_returns_error_key_on_total_failure(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        reg = _fresh_registry()
        reg._metadata.clear()
        adapter = CopilotModelAdapter(registry=reg, default_model="gone")
        response = adapter.process({"prompt": "test"})
        assert "error" in response


# ===========================================================================
# 8. Performance consistency
# ===========================================================================

class TestPerformanceConsistency:
    def test_unified_stack_returns_deterministic_limb_count(self):
        reg = _fresh_registry()
        model = reg.load("unified-stack")
        assert model is not None
        states = [0.1] * 8
        r1 = model.forward(states, task_signal="perf test")
        r2 = model.forward(states, task_signal="perf test")
        assert len(r1["limb_states"]) == len(r2["limb_states"]) == 8

    def test_adapter_latency_is_positive(self):
        from integration.copilot_model_adapter import CopilotModelAdapter
        adapter = CopilotModelAdapter(registry=_fresh_registry(), default_model="unified-stack")
        response = adapter.process({"prompt": "latency test", "model": "unified-stack"})
        assert response["latency_ms"] >= 0.0


# ===========================================================================
# 9. Capability matching (task domain → best model)
# ===========================================================================

class TestCapabilityMatching:
    def test_find_by_capability_spatial(self):
        reg = _fresh_registry()
        names = reg.find_by_capability("spatial")
        assert "unified-stack" in names
        assert "unified-stack-16" in names
        assert "gpt-4" not in names

    def test_find_by_capability_multi_domain(self):
        reg = _fresh_registry()
        names = reg.find_by_capability("multi-domain")
        assert "unified-stack-16" in names
        assert "unified-stack" not in names

    def test_find_by_capability_reasoning_includes_all(self):
        reg = _fresh_registry()
        names = reg.find_by_capability("reasoning")
        assert len(names) >= 4


# ===========================================================================
# 10. Version syntax
# ===========================================================================

class TestVersionSyntax:
    def test_16_limb_variant_resolves_to_unified_stack_16(self):
        reg = _fresh_registry()
        canonical = reg.resolve_name("unified-stack:16-limb")
        assert canonical == "unified-stack-16"

    def test_8_limb_variant_resolves_to_unified_stack(self):
        reg = _fresh_registry()
        canonical = reg.resolve_name("unified-stack:8-limb")
        assert canonical == "unified-stack"

    def test_v1_0_variant_resolves(self):
        reg = _fresh_registry()
        canonical = reg.resolve_name("unified-stack:v1.0")
        assert canonical == "unified-stack"

    def test_unknown_variant_falls_back_to_base(self):
        reg = _fresh_registry()
        # Should warn but resolve to base
        canonical = reg.resolve_name("unified-stack:v9.9")
        assert canonical == "unified-stack"

    def test_completely_unknown_spec_raises_value_error(self):
        reg = _fresh_registry()
        with pytest.raises(ValueError):
            reg.resolve_name("totally-fake-model:weird-variant")


# ===========================================================================
# 11. Model listing / discovery
# ===========================================================================

class TestModelListingDiscovery:
    def test_list_models_returns_at_least_four(self):
        reg = _fresh_registry()
        models = reg.list_models()
        assert len(models) >= 4

    def test_list_models_includes_required_models(self):
        reg = _fresh_registry()
        names = {m.name for m in reg.list_models()}
        required = {"unified-stack", "unified-stack-16", "gpt-4", "claude-3-opus"}
        assert required.issubset(names)

    def test_cli_list_models_json_output_is_valid_json(self, capsys):
        from cli_model_selector import main
        rc = main(["--list-models", "--json-output"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) >= 4

    def test_validate_model_returns_zero_for_known_model(self, capsys):
        from cli_model_selector import main
        rc = main(["--validate-model", "unified-stack"])
        assert rc == 0

    def test_validate_model_returns_nonzero_for_unknown(self, capsys):
        from cli_model_selector import main
        rc = main(["--validate-model", "fake-model-xyz"])
        assert rc != 0
