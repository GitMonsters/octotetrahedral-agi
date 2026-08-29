import json
import re

import pyarrow.parquet as pq

OUT = "data/combined_train.jsonl"
SOURCES = [
    "training_data.jsonl",
    "clarin_enriched_data.jsonl",
    "data/transcripts.jsonl",
]
EVAL = "data/eval_heldout.jsonl"
WIKI_PARQUET = "/tmp/wikitext2.parquet"

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[-])')
MIN_WORDS = 3
MAX_WORDS = 128

def clean_wiki(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"==?\s*.*?\s*=+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if MIN_WORDS <= len(s.split()) <= MAX_WORDS]

def main():
    with open(EVAL) as f:
        eval_set = {json.loads(l).get("text", "") for l in f}

    seen = set()
    counts = {}
    with open(OUT, "w") as out:
        for fp in SOURCES:
            n = 0
            with open(fp) as f:
                for line in f:
                    entry = json.loads(line)
                    text = entry.get("text") or " ".join(entry.get("tokens", []))
                    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
                        s = s.strip()
                        words = s.split()
                        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                            continue
                        if s in seen or s in eval_set:
                            continue
                        seen.add(s)
                        out.write(json.dumps({"text": s}) + "\n")
                        n += 1
            counts[fp] = n
            print(f"{fp}: +{n}")

        t = pq.read_table(WIKI_PARQUET).to_pandas()
        n = 0
        for text in t["text"].tolist():
            for s in clean_wiki(text):
                words = s.split()
                if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                    continue
                if s in seen or s in eval_set:
                    continue
                seen.add(s)
                out.write(json.dumps({"text": s}) + "\n")
                n += 1
        counts["wikitext-2-raw"] = n
        print(f"wikitext-2-raw: +{n}")

    total = sum(counts.values())
    print(f"\nTOTAL: {total} sentences -> {OUT}")

if __name__ == "__main__":
    main()