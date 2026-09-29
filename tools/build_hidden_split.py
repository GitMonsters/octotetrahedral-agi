#!/usr/bin/env python3
"""Build the SEALED hidden-eval split: data/eval_hidden.jsonl.

The training set (data/combined_train.jsonl) already contains wikitext-2-raw
minus the eval-heldout sentences, so any further wikitext would leak. This
split is drawn from corpus documents the LM has never trained on --
theory_of_everything and the instruction files -- which the retrieval chat
serves verbatim. Sentences are deduplicated against BOTH the training set and
data/eval_heldout.jsonl, so nothing here can have been memorised during
training. Treat this file as read-only at validation time: never score it
during development or model selection (mirrors Weco's hidden final exam).
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "eval_hidden.jsonl"
SOURCES = [
    REPO / "data" / "theory_of_everything.jsonl",
    REPO / "data" / "instructions.jsonl",
    REPO / "data" / "instructions_v2.jsonl",
]
EXCLUDE = [REPO / "data" / "combined_train.jsonl", REPO / "data" / "eval_heldout.jsonl"]
MAX_SENTS = 500
MIN_WORDS, MAX_WORDS = 3, 128

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[-])")


def load_text(path):
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get("text") or " ".join(entry.get("tokens", []))
            if text:
                yield text


def collect_excluded():
    known = set()
    for fp in EXCLUDE:
        if fp.exists():
            for text in load_text(fp):
                for s in re.split(r"(?<=[.!?])\s+", text.strip()):
                    known.add(s.strip())
    return known


def main():
    excluded = collect_excluded()
    print(f"excluded sentences: {len(excluded)}")
    picked = []
    seen = set()
    for fp in SOURCES:
        n = 0
        for text in load_text(fp):
            for s in SENT_SPLIT.split(text):
                s = s.strip()
                words = s.split()
                if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                    continue
                if s in excluded or s in seen:
                    continue
                if sum(len(w) for w in words) > 400:
                    continue
                seen.add(s)
                picked.append({"text": s})
                n += 1
        print(f"{fp.name}: +{n}")
    picked = picked[:MAX_SENTS]
    print(f"total: {len(picked)} -> {OUT}")
    with open(OUT, "w") as f:
        for entry in picked:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()