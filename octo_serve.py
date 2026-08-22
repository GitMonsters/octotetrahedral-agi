"""
OctoTetrahedral Server — POS Tagger + Language Model

Pure OctoTetrahedral architecture:
  - BiLSTM backbone (99.6% POS accuracy)
  - POS head (frozen at 99.6%)
  - LM head (text generation, ppl 1.18)
  - TranscendPlexity, WorkingMemory, Reservoir, Cohesion

Usage:
    python octo_serve.py
    python octo_serve.py --port 9000
"""

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
tagger = None
char_vocab = None
word_vocab = None
device = None
tag_inv = {}
POS_VOCAB = {}
CHAR_PAD = 0

# Dual-head model for generation
dual_model = None
dual_char_vocab = None
dual_word_vocab = None
dual_inv_vocab = {}
BOS_ID = 2

# Transformer model (better generation)
transformer_model = None
transformer_char_vocab = None
transformer_word_vocab = None
transformer_inv_vocab = {}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TagRequest(BaseModel):
    text: str = Field(..., description="Text to POS-tag")
    analyze: bool = Field(False, description="Include full module diagnostics")


class BatchTagRequest(BaseModel):
    sentences: list[str] = Field(..., description="Sentences to POS-tag")
    analyze: bool = Field(False, description="Include full module diagnostics")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to complete")
    max_tokens: int = Field(50, description="Max tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(20, description="Top-k sampling")
    rep_penalty: float = Field(3.0, description="Repetition penalty (additive)")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    max_tokens: int = Field(50, description="Max tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(20, description="Top-k sampling")
    rep_penalty: float = Field(3.0, description="Repetition penalty (additive)")
    history: list[str] = Field(default_factory=list, description="Previous messages")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(target_device: str):
    global tagger, char_vocab, word_vocab, device, tag_inv, POS_VOCAB
    global dual_model, dual_char_vocab, dual_word_vocab, dual_inv_vocab
    global transformer_model, transformer_char_vocab, transformer_word_vocab, transformer_inv_vocab

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_pos_bilstm import (
        OctoTetrahedralPosTagger, CHAR_PAD as _CP, POS_VOCAB as _PV
    )
    CHAR_PAD_local = _CP
    POS_VOCAB = _PV
    tag_inv = {v: k for k, v in POS_VOCAB.items()}

    device = torch.device(target_device)

    ckpt_path = Path("checkpoints/octo_integrated_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/octo_pos_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("No POS tagger checkpoint found")

    logger.info(f"Loading OctoTetrahedral tagger from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    char_vocab = ckpt["char_vocab"]
    word_vocab = ckpt["word_vocab"]
    config = ckpt["config"]

    tagger = OctoTetrahedralPosTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_emb=config["char_emb"],
        word_emb=config["word_emb"],
        hidden_dim=config["hidden"],
        dropout=config["dropout"],
        max_loops=config.get("max_loops", 3),
    ).to(device)

    state = ckpt.get("model", ckpt.get("model_state_dict", {}))
    tagger.load_state_dict(state, strict=False)
    tagger.eval()

    total = sum(p.numel() for p in tagger.parameters())
    acc = ckpt.get("accuracy", "?")
    logger.info(f"OctoTetrahedral tagger loaded: {total / 1e6:.1f}M params, accuracy={acc}")

    # Load dual-head model for generation
    dual_ckpt_path = Path("checkpoints/octo_dual_best.pt")
    if dual_ckpt_path.exists():
        logger.info(f"Loading dual-head model from {dual_ckpt_path}...")
        from train_lm import OctoDualHead
        dual_ckpt = torch.load(dual_ckpt_path, map_location="cpu")

        dual_word_vocab = dual_ckpt["word_vocab"]
        dual_char_vocab = dual_ckpt["char_vocab"]
        dual_inv_vocab = {v: k for k, v in dual_word_vocab.items()}

        dual_tagger = OctoTetrahedralPosTagger(
            char_vocab_size=len(dual_char_vocab),
            word_vocab_size=len(dual_word_vocab),
            char_emb=config["char_emb"],
            word_emb=config["word_emb"],
            hidden_dim=config["hidden"],
            dropout=config["dropout"],
            max_loops=config.get("max_loops", 3),
        ).to(device)
        dual_tagger.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", {})), strict=False)

        dual_model = OctoDualHead(dual_tagger).to(device)
        state = dual_ckpt.get("model", dual_ckpt.get("model_state_dict", {}))
        dual_model.load_state_dict(state, strict=False)
        dual_model.eval()

        n_gen = sum(p.numel() for p in dual_model.lm_head.parameters())
        logger.info(f"Dual-head model loaded: LM head={n_gen / 1e6:.1f}M params, ppl={dual_ckpt.get('lm_ppl', '?')}")
    else:
        logger.warning("No octo_dual_best.pt — generation disabled")

    # Load transformer model (better generation quality)
    transformer_ckpt_path = Path("checkpoints/octo_transformer_best.pt")
    if transformer_ckpt_path.exists():
        logger.info(f"Loading transformer model from {transformer_ckpt_path}...")
        from train_transformer import OctoTransformerLM as OctoTransformer
        transformer_ckpt = torch.load(transformer_ckpt_path, map_location="cpu")

        transformer_word_vocab = transformer_ckpt["word_vocab"]
        transformer_char_vocab = transformer_ckpt["char_vocab"]
        transformer_inv_vocab = {v: k for k, v in transformer_word_vocab.items()}

        t_config = transformer_ckpt.get("config", {})
        transformer_model = OctoTransformer(
            word_vocab_size=len(transformer_word_vocab),
            char_vocab_size=len(transformer_char_vocab),
            d_model=t_config.get("d_model", 256),
            nhead=t_config.get("nhead", 8),
            num_layers=t_config.get("num_layers", 3),
            dim_ff=t_config.get("dim_ff", 512),
            dropout=t_config.get("dropout", 0.2),
        ).to(device)
        state = transformer_ckpt.get("model", {})
        transformer_model.load_state_dict(state, strict=False)
        transformer_model.eval()

        n_params = sum(p.numel() for p in transformer_model.parameters())
        logger.info(f"Transformer loaded: {n_params / 1e6:.1f}M params, ppl={transformer_ckpt.get('lm_ppl', '?')}")
    else:
        logger.warning("No octo_transformer_best.pt — transformer generation disabled")


# ---------------------------------------------------------------------------
# Tagging logic
# ---------------------------------------------------------------------------

def tag_words(words: list[str], max_word_len: int = 30):
    """Tag words using the OctoTetrahedral model."""
    word_ids = [word_vocab.get(w, 1) for w in words]
    char_ids = []
    for w in words:
        chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
        while len(chars) < max_word_len:
            chars.append(CHAR_PAD)
        char_ids.append(chars[:max_word_len])

    while len(word_ids) < len(words):
        word_ids.append(0)
    while len(char_ids) < len(words):
        char_ids.append([CHAR_PAD] * max_word_len)

    wid = torch.tensor([word_ids])
    cid = torch.tensor([char_ids])

    with torch.no_grad():
        out = tagger(wid, cid)

    pred_ids = out["pos_logits"][0].argmax(dim=-1).tolist()
    tags = [tag_inv.get(p, "_") for p in pred_ids[:len(words)]]

    tp = out.get("tp_state")
    stability = float(getattr(tp, "stability", 0)) if tp else 0
    if isinstance(stability, torch.Tensor):
        stability = float(stability.item())
    comp_loss = float(getattr(tp, "compounding_loss", 0)) if tp else 0
    if isinstance(comp_loss, torch.Tensor):
        comp_loss = float(comp_loss.item())

    result = {
        "words": words,
        "tags": tags,
        "paired": list(zip(words, tags)),
        "tag_distribution": dict(Counter(tags)),
    }

    if tagger is not None:
        diag = tagger.get_diagnostics()
        result["tp"] = {
            "phase": diag.get("phase", "UNKNOWN"),
            "stability": diag.get("stability", 0),
            "compounding_loss": diag.get("compounding_loss", 0),
            "alpha": diag.get("alpha", []),
            "cohesion": diag.get("cohesion", 0),
            "cohesion_history": diag.get("cohesion_history", []),
            "modules": diag.get("modules", {}),
        }

    return result


def extract_entities(paired: list[tuple[str, str]]) -> list[dict]:
    """Extract named entities from POS-tagged words."""
    entities = []
    current = []
    for word, tag in paired:
        if tag == "PROPN":
            current.append(word)
        else:
            if current:
                entities.append({
                    "text": " ".join(current),
                    "type": "ENTITY",
                    "words": current,
                })
                current = []
    if current:
        entities.append({
            "text": " ".join(current),
            "type": "ENTITY",
            "words": current,
        })
    return entities


def extract_keywords(paired: list[tuple[str, str]]) -> list[dict]:
    """Extract keywords (nouns and proper nouns)."""
    keywords = []
    for word, tag in paired:
        if tag in ("NOUN", "PROPN"):
            keywords.append({"word": word, "pos": tag})
    return keywords


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_args = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _args
    if _args is None:
        _args = parse_args()
    target = _args.device or "cpu"
    load_model(target)
    yield


app = FastAPI(
    title="OctoTetrahedral API",
    description="BiLSTM POS Tagger + TranscendPlexity + WorkingMemory + Reservoir + Cohesion",
    version="2.0.0",
    lifespan=lifespan,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OctoTetrahedral Server")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive dashboard."""
    html_path = Path(__file__).parent / "clarin_demo.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>OctoTetrahedral Server</h1><p>clarin_demo.html not found</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tagger is not None,
        "dual_model_loaded": dual_model is not None,
        "transformer_loaded": transformer_model is not None,
        "device": str(device),
        "architecture": "BiLSTM + OctoTetrahedral modules + Transformer LM",
        "modules": {
            "transcendplexity": True,
            "working_memory": True,
            "reservoir": True,
            "cohesion": True,
            "compound_loop": True,
        },
    }


@app.post("/tag")
async def tag(req: TagRequest):
    """POS-tag text with optional module diagnostics."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    words = req.text.split()
    result = tag_words(words)

    response = {
        "text": req.text,
        "words": result["words"],
        "pos_tags": result["tags"],
        "tag_distribution": result["tag_distribution"],
    }

    if req.analyze:
        response["tp"] = result["tp"]
        response["entities"] = extract_entities(result["paired"])
        response["keywords"] = extract_keywords(result["paired"])

    return response


@app.post("/tag/batch")
async def tag_batch(req: BatchTagRequest):
    """Batch POS-tag multiple sentences."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    results = []
    for text in req.sentences:
        words = text.split()
        result = tag_words(words)
        entry = {
            "text": text,
            "words": result["words"],
            "pos_tags": result["tags"],
            "tag_distribution": result["tag_distribution"],
        }
        if req.analyze:
            entry["tp"] = result["tp"]
            entry["entities"] = extract_entities(result["paired"])
            entry["keywords"] = extract_keywords(result["paired"])
        results.append(entry)

    return {"results": results, "count": len(results)}


@app.post("/analyze")
async def analyze(req: TagRequest):
    """Full linguistic analysis with all module diagnostics."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    words = req.text.split()
    result = tag_words(words)
    paired = result["paired"]
    tags = result["tags"]
    dist = result["tag_distribution"]
    tp = result["tp"]

    # Grammar patterns
    bigrams = [f"{tags[i]}→{tags[i+1]}" for i in range(len(tags) - 1)]

    # Noun phrases
    noun_phrases = []
    i = 0
    while i < len(tags):
        if tags[i] == "DET":
            j = i + 1
            while j < len(tags) and tags[j] in ("ADJ", "NUM"):
                j += 1
            if j < len(tags) and tags[j] == "NOUN":
                noun_phrases.append(" ".join(words[i:j+1]))
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    # POS profile
    all_tags = list(POS_VOCAB.values())
    profile = {t: dist.get(t, 0) for t in all_tags}

    # Complexity score
    noun_pct = dist.get("NOUN", 0) + dist.get("PROPN", 0)
    verb_pct = dist.get("VERB", 0) + dist.get("AUX", 0)
    adj_pct = dist.get("ADJ", 0)
    adv_pct = dist.get("ADV", 0)
    unique_pos = len(set(tags))
    complexity = (adj_pct + adv_pct) * 0.3 + unique_pos * 0.5 + tp.get("cohesion", 0) * 2

    return {
        "text": req.text,
        "words": words,
        "pos_tags": tags,
        "tag_distribution": dist,
        "entities": extract_entities(paired),
        "keywords": extract_keywords(paired),
        "noun_phrases": noun_phrases,
        "bigrams": bigrams,
        "pos_profile": profile,
        "complexity": {
            "score": round(complexity, 2),
            "label": "Simple" if complexity < 8 else "Moderate" if complexity < 12 else "Complex",
            "noun_pct": noun_pct,
            "verb_pct": verb_pct,
            "adj_pct": adj_pct,
            "adv_pct": adv_pct,
            "unique_pos": unique_pos,
        },
        "tp": tp,
    }


@app.get("/modules")
async def modules():
    """Get current state of all OctoTetrahedral modules."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    diag = tagger.get_diagnostics()
    return diag


@app.post("/reset")
async def reset():
    """Reset all module states (WorkingMemory, Cohesion, TP)."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    tagger.reset_state()
    return {"status": "reset", "modules_cohesion_reset": True}


@app.get("/data-stats")
async def data_stats():
    """Return stats about clarin_enriched_data.jsonl."""
    data_path = Path("clarin_enriched_data.jsonl")
    if not data_path.exists():
        return {"error": "clarin_enriched_data.jsonl not found"}

    tag_dist = Counter()
    total = 0
    lengths = []

    with open(data_path) as f:
        for line in f:
            total += 1
            entry = json.loads(line)
            tokens = entry.get("tokens", [])
            pos_tags = entry.get("pos_tags", [])
            lengths.append(len(tokens))
            for tag in pos_tags:
                tag_dist[tag] += 1

    total_tags = sum(tag_dist.values())
    return {
        "total_sentences": total,
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "tag_distribution": {t: {"count": c, "pct": round(c / total_tags * 100, 1)}
                             for t, c in tag_dist.most_common()},
    }


@app.get("/accuracy")
async def accuracy(sentences: int = 100):
    """Run POS accuracy benchmark."""
    if tagger is None:
        raise HTTPException(503, "Model not loaded")

    data_path = Path("clarin_enriched_data.jsonl")
    if not data_path.exists():
        return {"error": "clarin_enriched_data.jsonl not found"}

    entries = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= sentences:
                break
            entries.append(json.loads(line))

    correct = 0
    total = 0
    tag_correct = Counter()
    tag_total = Counter()

    for entry in entries:
        words = entry.get("tokens", [])
        pos_tags = entry.get("pos_tags", [])
        if not words:
            continue

        result = tag_words(words)
        for word_idx in range(min(len(words), len(pos_tags), len(result["tags"]))):
            pred = result["tags"][word_idx]
            gold = pos_tags[word_idx]
            total += 1
            tag_total[gold] += 1
            if pred == gold:
                correct += 1
                tag_correct[gold] += 1

    accuracy = correct / total if total > 0 else 0
    per_tag = {}
    for tag in sorted(tag_total.keys(), key=lambda t: -tag_total[t]):
        per_tag[tag] = {
            "accuracy": round(tag_correct[tag] / tag_total[tag], 4) if tag_total[tag] > 0 else 0,
            "support": tag_total[tag],
        }

    return {
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_tag": per_tag,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate text using the best available model (transformer > LSTM)."""
    use_transformer = transformer_model is not None
    use_dual = dual_model is not None

    if not use_transformer and not use_dual:
        raise HTTPException(503, "No generation model loaded")

    words = req.prompt.split()
    if not words:
        words = ["the"]

    if use_transformer:
        w_vocab = transformer_word_vocab
        c_vocab = transformer_char_vocab
        inv = transformer_inv_vocab
        model = transformer_model
        model_type = "transformer"
    else:
        w_vocab = dual_word_vocab
        c_vocab = dual_char_vocab
        inv = dual_inv_vocab
        model = dual_model
        model_type = "lstm"

    seed_ids = torch.tensor([[BOS_ID] + [w_vocab.get(w.lower(), 1) for w in words]])
    max_word_len = 30
    bos_chars = [c_vocab.get(c, 1) for c in "<bos>"[:max_word_len]]
    while len(bos_chars) < max_word_len:
        bos_chars.append(CHAR_PAD)
    seed_chars = torch.zeros(1, len(words) + 1, max_word_len, dtype=torch.long)
    seed_chars[0, 0] = torch.tensor(bos_chars[:max_word_len])
    for i, w in enumerate(words):
        chars = [c_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
        while len(chars) < max_word_len:
            chars.append(CHAR_PAD)
        seed_chars[0, i + 1] = torch.tensor(chars[:max_word_len])

    with torch.no_grad():
        gen_ids = model.generate(
            seed_ids.to(device), seed_chars.to(device),
            max_new=req.max_tokens, temperature=req.temperature, top_k=req.top_k,
            rep_penalty=req.rep_penalty,
        )

    gen_words = [inv.get(i.item(), "?") for i in gen_ids[0]]
    gen_words = [w for w in gen_words[len(words)+1:] if w not in ("<UNK>", "?", "<PAD>")]
    generated = " ".join(gen_words)

    tp_phase = "UNKNOWN"
    if hasattr(model, "_tp_state") and model._tp_state:
        tp_phase = getattr(model._tp_state, "phase_name", "UNKNOWN")

    return {
        "prompt": req.prompt,
        "generated": generated,
        "full_text": req.prompt + " " + generated,
        "words": gen_words,
        "tp_phase": tp_phase,
        "model": model_type,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat with the OctoTetrahedral model."""
    use_transformer = transformer_model is not None
    use_dual = dual_model is not None

    if not use_transformer and not use_dual:
        raise HTTPException(503, "No generation model loaded")

    prompt_words = req.message.split()
    if not prompt_words:
        prompt_words = ["hello"]

    if use_transformer:
        w_vocab = transformer_word_vocab
        c_vocab = transformer_char_vocab
        inv = transformer_inv_vocab
        model = transformer_model
        model_type = "transformer"
    else:
        w_vocab = dual_word_vocab
        c_vocab = dual_char_vocab
        inv = dual_inv_vocab
        model = dual_model
        model_type = "lstm"

    seed_ids = torch.tensor([[BOS_ID] + [w_vocab.get(w.lower(), 1) for w in prompt_words]])
    max_word_len = 30
    bos_chars = [c_vocab.get(c, 1) for c in "<bos>"[:max_word_len]]
    while len(bos_chars) < max_word_len:
        bos_chars.append(CHAR_PAD)
    seed_chars = torch.zeros(1, len(prompt_words) + 1, max_word_len, dtype=torch.long)
    seed_chars[0, 0] = torch.tensor(bos_chars[:max_word_len])
    for i, w in enumerate(prompt_words):
        chars = [c_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
        while len(chars) < max_word_len:
            chars.append(CHAR_PAD)
        seed_chars[0, i + 1] = torch.tensor(chars[:max_word_len])

    with torch.no_grad():
        gen_ids = model.generate(
            seed_ids.to(device), seed_chars.to(device),
            max_new=req.max_tokens, temperature=req.temperature, top_k=req.top_k,
            rep_penalty=req.rep_penalty,
        )

    gen_words = [inv.get(i.item(), "?") for i in gen_ids[0]]
    reply = " ".join([w for w in gen_words[len(prompt_words)+1:] if w not in ("<UNK>", "?", "<PAD>")])

    pos_result = tag_words(req.message.split())
    tp_phase = "UNKNOWN"
    if hasattr(model, "_tp_state") and model._tp_state:
        tp_phase = getattr(model._tp_state, "phase_name", "UNKNOWN")

    return {
        "user": req.message,
        "reply": reply,
        "pos_tags": pos_result["tags"],
        "tp_phase": tp_phase,
        "cohesion": pos_result["tp"].get("cohesion", 0),
        "model": model_type,
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = parse_args()
    uvicorn.run(app, host=_args.host, port=_args.port)
