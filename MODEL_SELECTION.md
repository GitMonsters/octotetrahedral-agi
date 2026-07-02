# Model Selection Guide

Complete guide for selecting and configuring the unified cognitive stack and
external models across all Copilot interfaces.

---

## Table of Contents

1. [Terminal / CLI](#terminal--cli)
2. [Chat UI](#chat-ui)
3. [Configuration File](#configuration-file)
4. [Available Models](#available-models)
5. [Model Switching Mid-Conversation](#model-switching-mid-conversation)
6. [Coherence Stats](#coherence-stats)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Adding Custom Models](#adding-custom-models)

---

## Terminal / CLI

### Basic usage

```bash
# Use the unified cognitive stack (8-limb)
python cli_model_selector.py --model unified-stack --task "explain recursion"

# Use the 16-limb extended model
python cli_model_selector.py --model unified-stack:16-limb --task "plan a multi-step route"

# Use GPT-4 (external)
python cli_model_selector.py --model gpt-4 --task "summarise this document"

# Use Claude 3 Opus (external)
python cli_model_selector.py --model claude-3-opus --task "review this code"
```

### Model specification syntax

| Syntax | Description |
|--------|-------------|
| `unified-stack` | 8-limb production model |
| `unified-stack:16-limb` | 16-limb experimental variant |
| `unified-stack:v1.0` | Pinned version |
| `gpt-4` | OpenAI GPT-4 |
| `claude-3-opus` | Anthropic Claude 3 Opus |

### List available models

```bash
python cli_model_selector.py --list-models

# JSON output for scripting
python cli_model_selector.py --list-models --json-output
```

### Validate a model specification

```bash
python cli_model_selector.py --validate-model unified-stack:16-limb
```

### Display model stats in output

```bash
python cli_model_selector.py --model unified-stack --task "reasoning task" --show-stats
```

Output example:
```
────────────────────────────────────────────────────────────
 Task   : reasoning task
  Model         : unified-stack — 8-limb quantum-biological unified model (production)
  Provider      : local
  Capabilities  : reasoning, language, spatial, planning
  Limbs         : 8
  Coherence     : 0.9823
  Coupling      : 0.8500
  Action channel: limb 3
  Latency       : 0.15 ms
  Limbs active  : 7/8
────────────────────────────────────────────────────────────
```

### Persist model selection

By default the last-used model is saved to `.copilot/last_model.json`.
Subsequent runs without `--model` reuse the saved selection.

```bash
# Run once with explicit model
python cli_model_selector.py --model unified-stack --task "first task"

# Subsequent runs reuse unified-stack automatically
python cli_model_selector.py --task "second task"

# Disable persistence for a single run
python cli_model_selector.py --model gpt-4 --task "one-off task" --no-persist
```

### Copilot task integration

Add `--model` to any copilot task command:

```bash
copilot task --model unified-stack --repo owner/repo --problem "implement feature X"
```

---

## Chat UI

### Adding the ModelSelector component

```tsx
import { ModelSelector } from "./components/ModelSelector";

function ChatWindow() {
  const handleModelChange = (model) => {
    console.log("Switched to:", model.id);
    // Notify your chat engine of the model change
  };

  return (
    <div>
      <ModelSelector onModelChange={handleModelChange} />
      {/* ... rest of chat UI */}
    </div>
  );
}
```

### Pushing real-time stats to the selector

```tsx
import { ModelSelector } from "./components/ModelSelector";

function ChatWindow({ lastInferenceStats }) {
  return (
    <ModelSelector stats={lastInferenceStats} />
  );
}
```

Where `lastInferenceStats` has shape:
```ts
{
  coherence: 0.982,    // 0–1
  latency: 0.15,       // ms
  limbsActive: 7,      // count
  actionChannel: 3,    // limb index
}
```

### Using the model store directly

```ts
import { modelStore } from "./stores/modelStore";

// Read current model
console.log(modelStore.current.id);  // "unified-stack"

// Switch model programmatically
modelStore.setModel("claude-3-opus");

// Subscribe to changes
const unsubscribe = modelStore.subscribe((model) => {
  console.log("Model changed to:", model.name);
});

// Cleanup
unsubscribe();

// Find models by capability
const planners = modelStore.findByCapability("planning");
```

---

## Configuration File

Create or edit `.copilot/config.yml` in your repository root:

```yaml
# Default model for all Copilot tasks
default_model: unified-stack

# Fallback chain (first available model wins)
fallback_chain:
  - unified-stack
  - gpt-4
  - claude-3-opus

models:
  unified-stack:
    description: "8-limb quantum-biological unified model (production)"
    limbs: 8
    coherence_threshold: 0.90
    batch_size: 32
    timeout_ms: 30000
    capabilities: [reasoning, language, spatial, planning]

  unified-stack-16:
    description: "16-limb extended model (experimental)"
    limbs: 16
    coherence_threshold: 0.92
    batch_size: 16
    timeout_ms: 50000
    capabilities: [reasoning, language, spatial, planning, multi-domain]

  gpt-4:
    description: "OpenAI GPT-4"
    batch_size: 64
    timeout_ms: 60000

  claude-3-opus:
    description: "Anthropic Claude 3 Opus"
    batch_size: 64
    timeout_ms: 60000

monitoring:
  enable_coherence_tracking: true
  enable_limb_profiling: true
  coherence_alert_threshold: 0.85
  collect_metrics: true

user_preferences:
  preferred_model: unified-stack
  show_model_stats: true
  auto_fallback: true
```

---

## Available Models

| Model | Limbs | Provider | Best For |
|-------|-------|----------|----------|
| `unified-stack` | 8 | local | General reasoning, language, spatial, planning |
| `unified-stack-16` | 16 | local | Multi-domain, complex compound tasks |
| `gpt-4` | — | OpenAI | Language, code generation, planning |
| `claude-3-opus` | — | Anthropic | Long context, analysis, reasoning |

### Capability matrix

| Model | reasoning | language | spatial | planning | multi-domain |
|-------|-----------|----------|---------|----------|--------------|
| unified-stack | ✅ | ✅ | ✅ | ✅ | ❌ |
| unified-stack-16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-4 | ✅ | ✅ | ❌ | ✅ | ❌ |
| claude-3-opus | ✅ | ✅ | ❌ | ✅ | ❌ |

---

## Model Switching Mid-Conversation

Switching models during a session is supported at both the CLI and UI level:

**CLI** — simply pass a different `--model` flag to each invocation.

**Chat UI** — use the dropdown to switch at any time.  The new model takes
effect immediately for the next message; previous turns are unaffected.

**Programmatic** (Python):

```python
from integration.copilot_integration import bootstrap

integration = bootstrap()

# First request uses unified-stack (from config)
response1 = integration.process_request({"prompt": "explain gravity"})

# Override mid-conversation
response2 = integration.process_request({
    "prompt": "now plan a mission to Mars",
    "model": "unified-stack-16",
})
```

---

## Coherence Stats

When using the unified cognitive stack, each inference returns coherence metadata:

```json
{
  "model": "unified-stack",
  "coherence": 0.9823,
  "action_channel": 3,
  "limb_metadata": {
    "limb_states": [0.82, 0.91, 0.77, 0.95, 0.88, 0.73, 0.90, 0.85],
    "active_limbs": 7,
    "dominant_limb": 3,
    "coupling_strength": 0.85
  },
  "latency_ms": 0.15
}
```

- **coherence** ≥ 0.90 → nominal operation
- **coherence** < 0.85 → alert threshold (configure in `monitoring.coherence_alert_threshold`)
- **action_channel** → dominant limb index (0–7 for 8-limb model)

---

## Performance Tuning

| Goal | Recommendation |
|------|---------------|
| Lowest latency | `unified-stack` (local, sub-ms) |
| Highest coherence | `unified-stack-16` (0.92 threshold) |
| Large batch jobs | `gpt-4` or `claude-3-opus` (batch_size 64) |
| Multi-domain tasks | `unified-stack-16` |
| Offline / no API key | `unified-stack` or `unified-stack-16` |

Tunable parameters in `config.yml`:

```yaml
models:
  unified-stack:
    batch_size: 32        # Reduce for lower memory; increase for throughput
    timeout_ms: 30000     # Adjust per environment
    coherence_threshold: 0.90  # Raise for stricter quality gate
```

---

## Troubleshooting

### Model not found

```
ValueError: Unknown model specification: 'my-custom-model'
```

→ Check `--list-models` output.  Register custom models in `.copilot/config.yml`.

### Fallback triggered unexpectedly

```
WARNING: Model 'unified-stack' not available; falling back to 'gpt-4'.
```

→ Ensure the `unified` package is installed and importable:
```bash
python -c "from unified.forward_model import UnifiedForwardModel; print('OK')"
```

### Coherence below threshold

→ Try `unified-stack-16` for higher coherence, or reduce task complexity.
→ Check `monitoring.coherence_alert_threshold` in config.

### Config file not loaded

→ Ensure `.copilot/config.yml` is in the repository root (same directory as
   where you run your commands).
→ Install PyYAML: `pip install pyyaml`

### Chat UI: model selection not persisting

→ Confirm `localStorage` is available in your browser.  The store falls back
   to in-memory state if `localStorage` is unavailable (e.g. private browsing).

---

## Best Practices

| Task Type | Recommended Model |
|-----------|------------------|
| Code generation | `gpt-4` or `claude-3-opus` |
| Spatial / visual reasoning | `unified-stack` or `unified-stack-16` |
| Multi-step planning | `unified-stack` |
| Long document analysis | `claude-3-opus` |
| Compound multi-domain | `unified-stack-16` |
| Offline / air-gapped | `unified-stack` |

- Always configure a `fallback_chain` in `.copilot/config.yml` to prevent
  hard failures when a model is unavailable.
- Enable `monitoring.collect_metrics: true` in production to track coherence
  trends over time.
- Use `unified-stack:16-limb` (or `unified-stack-16`) for multi-domain tasks
  that span reasoning, spatial, and planning domains simultaneously.

---

## Adding Custom Models

### Via config file

```yaml
models:
  my-fine-tuned-model:
    description: "Fine-tuned variant for legal document analysis"
    limbs: 0
    batch_size: 32
    timeout_ms: 45000
    capabilities: [reasoning, language]
```

### Via Python API

```python
from model_registry import ModelMetadata, get_registry

registry = get_registry()
registry.register(
    ModelMetadata(
        name="my-fine-tuned-model",
        description="Fine-tuned variant for legal document analysis",
        capabilities=["reasoning", "language"],
        batch_size=32,
        timeout_ms=45000,
    ),
    loader=lambda: MyCustomModel(),
)
```

Custom models registered this way participate in the fallback chain and can be
selected via `--model my-fine-tuned-model` on the CLI or via the chat UI
dropdown (after adding to `AVAILABLE_MODELS` in `stores/modelStore.ts`).
