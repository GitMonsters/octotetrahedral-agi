/**
 * components/ModelSelector.tsx
 *
 * Dropdown component for Copilot chat window.
 * Shows available models, real-time stats, supports mid-conversation switching,
 * and displays a visual indicator when using the Unified Cognitive Stack.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { modelStore, ModelInfo, ModelStats } from "../stores/modelStore";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface StatsRowProps {
  label: string;
  value: string | number;
}

function StatsRow({ label, value }: StatsRowProps): React.ReactElement {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#888" }}>
      <span>{label}</span>
      <span style={{ fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// UnifiedStackBadge
// ---------------------------------------------------------------------------

function UnifiedStackBadge(): React.ReactElement {
  return (
    <span
      title="Unified Cognitive Stack active"
      style={{
        display: "inline-block",
        background: "linear-gradient(135deg, #6c5ce7, #00cec9)",
        color: "#fff",
        borderRadius: "3px",
        padding: "1px 5px",
        fontSize: "0.65rem",
        fontWeight: 700,
        letterSpacing: "0.04em",
        marginLeft: "6px",
        verticalAlign: "middle",
      }}
    >
      ⬡ UCS
    </span>
  );
}

// ---------------------------------------------------------------------------
// ModelOption (single item in dropdown)
// ---------------------------------------------------------------------------

interface ModelOptionProps {
  model: ModelInfo;
  isSelected: boolean;
  onSelect: (model: ModelInfo) => void;
}

function ModelOption({ model, isSelected, onSelect }: ModelOptionProps): React.ReactElement {
  return (
    <div
      role="option"
      aria-selected={isSelected}
      onClick={() => onSelect(model)}
      style={{
        padding: "8px 12px",
        cursor: "pointer",
        background: isSelected ? "#f0eeff" : "transparent",
        borderLeft: isSelected ? "3px solid #6c5ce7" : "3px solid transparent",
      }}
    >
      <div style={{ fontWeight: 600, fontSize: "0.875rem" }}>
        {model.name}
        {model.isUnifiedStack && <UnifiedStackBadge />}
      </div>
      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "2px" }}>
        {model.description}
      </div>
      {model.limbs > 0 && (
        <div style={{ fontSize: "0.7rem", color: "#a0a0a0", marginTop: "1px" }}>
          {model.limbs} limbs · {model.provider}
        </div>
      )}
      {model.stats && (
        <div style={{ marginTop: "4px" }}>
          <StatsRow label="Coherence" value={model.stats.coherence.toFixed(3)} />
          <StatsRow label="Latency" value={`${model.stats.latency.toFixed(1)} ms`} />
          {model.limbs > 0 && (
            <StatsRow label="Limbs active" value={`${model.stats.limbsActive}/${model.limbs}`} />
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ModelSelector
// ---------------------------------------------------------------------------

interface ModelSelectorProps {
  /** Called each time the user selects a model. */
  onModelChange?: (model: ModelInfo) => void;
  /** Optional live stats update callback (pass incoming stats here). */
  stats?: ModelStats;
}

export function ModelSelector({ onModelChange, stats }: ModelSelectorProps): React.ReactElement {
  const [current, setCurrent] = useState<ModelInfo>(modelStore.current);
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Subscribe to store changes
  useEffect(() => {
    const unsub = modelStore.subscribe((m) => setCurrent(m));
    return unsub;
  }, []);

  // Push incoming stats into store
  useEffect(() => {
    if (stats) {
      modelStore.updateStats(current.id, stats);
    }
  }, [stats, current.id]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleSelect = useCallback(
    (model: ModelInfo) => {
      modelStore.setModel(model);
      setOpen(false);
      onModelChange?.(model);
    },
    [onModelChange],
  );

  return (
    <div
      ref={dropdownRef}
      style={{ position: "relative", display: "inline-block", minWidth: "220px", fontFamily: "system-ui, sans-serif" }}
    >
      {/* Trigger button */}
      <button
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "6px 12px",
          border: "1px solid #ddd",
          borderRadius: "6px",
          background: "#fff",
          cursor: "pointer",
          width: "100%",
          justifyContent: "space-between",
        }}
      >
        <span>
          {current.name}
          {current.isUnifiedStack && <UnifiedStackBadge />}
        </span>
        <span style={{ fontSize: "0.7rem", color: "#888" }}>{open ? "▲" : "▼"}</span>
      </button>

      {/* Dropdown panel */}
      {open && (
        <div
          role="listbox"
          aria-label="Select model"
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            left: 0,
            right: 0,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "6px",
            boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
            zIndex: 1000,
            maxHeight: "360px",
            overflowY: "auto",
          }}
        >
          {modelStore.models.map((model) => (
            <ModelOption
              key={model.id}
              model={model}
              isSelected={model.id === current.id}
              onSelect={handleSelect}
            />
          ))}
        </div>
      )}

      {/* Inline stats strip (shown when a unified-stack model is active) */}
      {current.isUnifiedStack && current.stats && (
        <div
          style={{
            marginTop: "4px",
            padding: "4px 8px",
            background: "#f8f6ff",
            borderRadius: "4px",
            border: "1px solid #e0d8ff",
            fontSize: "0.72rem",
            color: "#666",
          }}
        >
          <StatsRow label="Coherence" value={current.stats.coherence.toFixed(3)} />
          <StatsRow label="Latency" value={`${current.stats.latency.toFixed(1)} ms`} />
          <StatsRow label="Limbs active" value={`${current.stats.limbsActive}/${current.limbs}`} />
        </div>
      )}
    </div>
  );
}

export default ModelSelector;
