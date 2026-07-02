/**
 * stores/modelStore.ts
 *
 * React/TypeScript store for chat-UI model state.
 * Persists selection to localStorage, emits change events, and exposes
 * model list + capabilities for the ModelSelector dropdown.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelCapabilities {
  reasoning: boolean;
  language: boolean;
  spatial: boolean;
  planning: boolean;
  multiDomain: boolean;
}

export interface ModelStats {
  coherence: number;    // 0–1
  latency: number;      // ms
  limbsActive: number;  // count
  actionChannel: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  provider: "local" | "openai" | "anthropic";
  limbs: number;
  capabilities: ModelCapabilities;
  isUnifiedStack: boolean;
  stats?: ModelStats;
}

export type ModelChangeHandler = (model: ModelInfo) => void;

// ---------------------------------------------------------------------------
// Available models catalog
// ---------------------------------------------------------------------------

const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: "unified-stack",
    name: "Unified Cognitive Stack",
    description: "8-limb quantum-biological unified model (production)",
    provider: "local",
    limbs: 8,
    capabilities: { reasoning: true, language: true, spatial: true, planning: true, multiDomain: false },
    isUnifiedStack: true,
  },
  {
    id: "unified-stack-16",
    name: "Unified Stack 16-Limb",
    description: "16-limb extended model (experimental)",
    provider: "local",
    limbs: 16,
    capabilities: { reasoning: true, language: true, spatial: true, planning: true, multiDomain: true },
    isUnifiedStack: true,
  },
  {
    id: "gpt-4",
    name: "GPT-4",
    description: "OpenAI GPT-4",
    provider: "openai",
    limbs: 0,
    capabilities: { reasoning: true, language: true, spatial: false, planning: true, multiDomain: false },
    isUnifiedStack: false,
  },
  {
    id: "claude-3-opus",
    name: "Claude 3 Opus",
    description: "Anthropic Claude 3 Opus",
    provider: "anthropic",
    limbs: 0,
    capabilities: { reasoning: true, language: true, spatial: false, planning: true, multiDomain: false },
    isUnifiedStack: false,
  },
];

// Default model when no saved preference exists.
// Matches the default_model in .copilot/config.yml so the CLI and chat UI
// agree on which model to use out of the box.
const STORAGE_KEY = "copilot_selected_model";
const DEFAULT_MODEL_ID = "unified-stack";

// ---------------------------------------------------------------------------
// ModelStore
// ---------------------------------------------------------------------------

class ModelStore {
  private _current: ModelInfo;
  private _handlers: Set<ModelChangeHandler> = new Set();

  constructor() {
    const saved = this._loadFromStorage();
    this._current = saved ?? this._findById(DEFAULT_MODEL_ID) ?? AVAILABLE_MODELS[0];
  }

  // ---- Getters ----

  get current(): ModelInfo {
    return this._current;
  }

  get models(): ModelInfo[] {
    return [...AVAILABLE_MODELS];
  }

  getById(id: string): ModelInfo | undefined {
    return this._findById(id);
  }

  findByCapability(cap: keyof ModelCapabilities): ModelInfo[] {
    return AVAILABLE_MODELS.filter((m) => m.capabilities[cap]);
  }

  // ---- Setters ----

  setModel(idOrModel: string | ModelInfo): void {
    const model =
      typeof idOrModel === "string" ? this._findById(idOrModel) : idOrModel;
    if (!model) {
      console.warn(`[ModelStore] Unknown model id: ${idOrModel}`);
      return;
    }
    this._current = model;
    this._saveToStorage(model.id);
    this._emit(model);
  }

  updateStats(id: string, stats: ModelStats): void {
    const idx = AVAILABLE_MODELS.findIndex((m) => m.id === id);
    if (idx !== -1) {
      AVAILABLE_MODELS[idx] = { ...AVAILABLE_MODELS[idx], stats };
    }
    if (this._current.id === id) {
      this._current = { ...this._current, stats };
      this._emit(this._current);
    }
  }

  // ---- Subscriptions ----

  subscribe(handler: ModelChangeHandler): () => void {
    this._handlers.add(handler);
    return () => this._handlers.delete(handler);
  }

  // ---- Persistence ----

  private _loadFromStorage(): ModelInfo | null {
    if (typeof localStorage === "undefined") return null;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw) as { id: string };
      return this._findById(parsed.id) ?? null;
    } catch {
      return null;
    }
  }

  private _saveToStorage(id: string): void {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ id, savedAt: Date.now() }));
  }

  // ---- Helpers ----

  private _findById(id: string): ModelInfo | undefined {
    return AVAILABLE_MODELS.find((m) => m.id === id);
  }

  private _emit(model: ModelInfo): void {
    this._handlers.forEach((h) => h(model));
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const modelStore = new ModelStore();
export type { ModelStore };
