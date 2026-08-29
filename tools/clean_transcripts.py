import json
import os
import re
import sys

SRC_DIRS = [
    "/Users/evanpieser/Downloads",
    "/Users/evanpieser/Downloads/subs",
]
DST = "data/transcripts.jsonl"
EVAL = "data/eval_heldout.jsonl"

ARTIFACT = re.compile(r"\[music\]|>>|\[.*?\]|\([^)]{0,12}\)")
DUP_WORD = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[-])')
MIN_WORDS = 4

def clean_text(raw):
    text = "\n".join(raw)
    for tok in ("\ufeff", "&amp;", "&quot;", "&lt;", "&gt;"):
        text = text.replace(tok, " ")
    text = text.replace("\t", " ")
    text = ARTIFACT.sub(" ", text)
    text = DUP_WORD.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    sents = []
    for s in SENT_SPLIT.split(text):
        s = s.strip().strip(",").strip()
        words = s.split()
        if MIN_WORDS <= len(words) <= 128:
            sents.append(s)
    return sents

def main():
    eval_set = set()
    if os.path.exists(EVAL):
        with open(EVAL) as f:
            for line in f:
                entry = json.loads(line)
                eval_set.add(entry.get("text", ""))

    files = []
    for d in SRC_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt") and ("DownSub" in fn or fn.startswith(("[English", "[English ]"))) or fn in ("subs/",):
                files.append(os.path.join(d, fn))
        # fallback: any DownSub txt deeper
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt") and "DownSub" in fn:
                files.append(os.path.join(d, fn))

    # dedupe by path and by file hash (DownSub re-downloads are identical)
    seen_hash = set()
    unique = []
    for fp in sorted(set(files)):
        if not os.path.exists(fp):
            continue
        h = os.path.getsize(fp)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        unique.append(fp)

    seen_sent = set()
    n_file = 0
    n_sent = 0
    n_skip = 0
    with open(DST, "w") as out:
        for fp in sorted(unique):
            with open(fp, errors="ignore") as f:
                sents = clean_text(f.read().splitlines())
            added = 0
            for s in sents:
                if s in seen_sent or s in eval_set:
                    n_skip += 1
                    continue
                seen_sent.add(s)
                out.write(json.dumps({"text": s}) + "\n")
                added += 1
            n_sent += added
            n_file += 1
            print(f"{os.path.basename(fp)[:60]:60s} +{added}")
    print(f"\nwrote {n_sent} unique sentences from {n_file} files -> {DST} (skipped {n_skip})")

if __name__ == "__main__":
    main()