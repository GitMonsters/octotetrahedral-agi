"""
Transcript → Skills extractor.

Ingests any educational/explanatory text (video transcripts, papers, notes) and
distills it into:
  • concept_index  — ranked named concepts (proper-noun phrases + repeated terms)
  • skill_seeds    — actionable skills the AGI could learn, mapped to limbs
  • module_idea    — a short "what could we build?" suggestion per concept
  • glossary       — beginner-friendly one-liners

Pure stdlib. No nltk, no sklearn, no torch.

Usage:
    python3 tools/transcript_skill_extractor.py path/to/transcript.txt
    python3 tools/transcript_skill_extractor.py path/to/transcript.txt --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ─── limb routing rules: keyword → limb ──────────────────────────────────────
LIMB_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "perception":     ("measure", "observ", "detect", "polariz", "spin",
                       "image", "see", "view", "signal", "photon"),
    "reasoning":      ("theorem", "proof", "inequal", "logic", "deriv",
                       "implies", "contradict", "argument", "infer"),
    "memory":         ("history", "record", "store", "previous", "recall"),
    "planning":       ("strategy", "plan", "schedul", "sequence", "step"),
    "action":         ("apply", "execute", "perform", "do ", "rotate", "transform"),
    "language":       ("describe", "explain", "say", "word", "sentence"),
    "metacognition":  ("review", "monitor", "evaluate", "critic",
                       "assumption", "paradox", "uncertain", "hidden variable"),
    "spatial":        ("vector", "direction", "axis", "plane", "geometry",
                       "rotation", "angle"),
}

STOPWORDS = {
    "the","a","an","of","to","and","or","is","are","was","were","be","been",
    "in","on","at","by","for","with","that","this","these","those","it","its",
    "as","but","if","then","so","not","no","do","does","did","have","has","had",
    "i","you","he","she","they","we","my","your","our","their","me","us","them",
    "what","which","who","whom","when","where","why","how","just","very","really",
    "all","any","some","one","two","three","also","like","get","got","go","going",
    "now","here","there","said","says","saying","because","about","into","from",
    "up","down","out","over","under","again","more","most","such","than","only",
    "even","much","many","every","ever","never","still","yet","first","last",
    "thing","things","kind","sort","way","time","whole","whole","right","okay",
    "let","lets","kind","stuff","actually","basically","gonna","wanna","gotta",
    "youre","its","im","thats","theres","youve","weve","theyre","cant","dont",
    "well","know","think","mean","sure","yes","yeah","oh","hey","music",
}

PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[\s\-][A-Z][a-z]{2,}){0,3})\b")
APOSTROPHE_RE  = re.compile(r"^[A-Za-z]+'[A-Za-z]+$")
WORD_RE        = re.compile(r"[a-zA-Z][a-zA-Z\-']{2,}")
SENT_RE        = re.compile(r"(?<=[.!?])\s+")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def proper_phrases(text: str) -> Counter:
    """Capitalised multi-word phrases (likely names/concepts)."""
    counts: Counter = Counter()
    for m in PROPER_NOUN_RE.findall(text):
        phrase = m.strip()
        if phrase.lower() in STOPWORDS:
            continue
        # Skip pure single-word sentence-starters by requiring length
        if len(phrase) < 4 or phrase.lower() in {"hey","music","okay","sure"}:
            continue
        counts[phrase] += 1
    return counts


def repeated_terms(text: str, min_count: int = 5) -> Counter:
    """Lowercase content words that recur."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    counts = Counter(words)
    return Counter({w: c for w, c in counts.items() if c >= min_count})


def assign_limb(term: str) -> str:
    t = term.lower()
    best, score = "metacognition", 0
    for limb, kws in LIMB_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in t)
        if s > score:
            best, score = limb, s
    return best


def find_definition(term: str, text: str, max_chars: int = 180) -> str:
    """Find a sentence that defines or first introduces the term."""
    needle = term.lower()
    for sent in SENT_RE.split(text):
        s = sent.strip().replace("\n", " ")
        if needle in s.lower() and len(s) > 20:
            if len(s) > max_chars:
                s = s[:max_chars - 1] + "…"
            return s
    return ""


def synthesise_skill(term: str, definition: str) -> Dict[str, str]:
    """Turn a concept into a candidate AGI skill row."""
    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")[:48]
    limb = assign_limb(term + " " + definition)
    return {
        "id":          f"learned-{slug}",
        "name":        term,
        "category":    "learned-concept",
        "description": definition or f"Concept '{term}' learned from transcript.",
        "source":      "transcript-extractor",
        "limb":        limb,
        "complexity":  3 if len(definition) > 80 else 2,
    }


def module_suggestion(skill: Dict[str, str]) -> str:
    """Plain-English 'what could we build?' suggestion."""
    limb = skill["limb"]
    name = skill["name"]
    templates = {
        "reasoning":     f"Add a `{name}`-style proof checker to the reasoning limb.",
        "metacognition": f"Use `{name}` as a self-doubt heuristic in the metacognition limb.",
        "perception":    f"Train a perception probe inspired by `{name}` measurements.",
        "memory":        f"Store `{name}` traces for later replay in the memory limb.",
        "planning":      f"Wire a `{name}` strategy into the planning limb.",
        "action":        f"Implement `{name}` as an executable transform.",
        "language":      f"Add a `{name}` explainer to the language limb.",
        "spatial":       f"Map `{name}` into the tetrahedral geometry module.",
    }
    return templates.get(limb, f"Investigate `{name}` for limb '{limb}'.")


def extract(text: str) -> Dict:
    proper = proper_phrases(text)
    common = repeated_terms(text)
    # Top concepts: union, ranked by count
    merged = Counter()
    for k, v in proper.items():
        merged[k] += v * 2  # weight proper-nouns higher
    for k, v in common.items():
        merged[k.title()] += v
    # Drop contractions like "It's", "We're"
    merged = Counter({k: v for k, v in merged.items()
                      if not APOSTROPHE_RE.match(k)})

    top = merged.most_common(20)

    skills: List[Dict] = []
    glossary: Dict[str, str] = {}
    modules: List[str] = []
    limb_dist: Counter = Counter()

    for term, _ in top:
        defn = find_definition(term, text)
        sk = synthesise_skill(term, defn)
        skills.append(sk)
        glossary[term] = defn or "(no inline definition found)"
        modules.append(module_suggestion(sk))
        limb_dist[sk["limb"]] += 1

    return {
        "stats": {
            "total_chars": len(text),
            "total_words": len(WORD_RE.findall(text)),
            "unique_concepts": len(merged),
        },
        "concept_index": [{"term": t, "weight": w} for t, w in top],
        "skill_seeds":   skills,
        "module_ideas":  modules,
        "glossary":      glossary,
        "limb_distribution": dict(limb_dist),
    }


def render_console(result: Dict) -> str:
    out = []
    s = result["stats"]
    out.append(f"📖 Analysed {s['total_words']:,} words → {s['unique_concepts']} candidate concepts\n")
    out.append("🏆 Top concepts:")
    for c in result["concept_index"][:10]:
        out.append(f"   • {c['term']:<32} weight={c['weight']}")
    out.append("\n🧠 Skill seeds (mapped to limbs):")
    for sk in result["skill_seeds"][:10]:
        out.append(f"   • [{sk['limb']:<13}] {sk['id']}")
    out.append("\n🔧 Module suggestions:")
    for m in result["module_ideas"][:10]:
        out.append(f"   • {m}")
    out.append("\n🧬 Limb distribution: " +
               ", ".join(f"{k}={v}" for k, v in result["limb_distribution"].items()))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--json", type=Path, help="write full JSON result")
    args = ap.parse_args()

    text = load_text(args.transcript)
    result = extract(text)

    print(render_console(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n💾 Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
