import json
import re
import sys

SRC = "/Users/evanpieser/Downloads/The Final Verdict On The Theory Of Everything [DownSub.com].txt"
DST = "data/theory_of_everything.jsonl"

ARTIFACT = re.compile(r"\[music\]|>>|\[.*?\]")
DUP_WORD = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[-])')

def clean_lines(raw):
    sentences = []
    buf = ""
    for line in raw:
        line = ARTIFACT.sub(" ", line).strip()
        if not line:
            continue
        if buf:
            if buf.endswith((".", "!", "?")) and line[0].isupper():
                buf += " " + line
    return buf

def main():
    with open(SRC) as f:
        raw = f.read().splitlines()
    text = " ".join(t for t in (ARTIFACT.sub(" ", l).strip() for l in raw) if t)
    text = DUP_WORD.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 3]
    with open(DST, "w") as f:
        for s in sents:
            f.write(json.dumps({"text": s}) + "\n")
    print(f"wrote {len(sents)} sentences -> {DST}")
    for s in sents[-6:]:
        print("  " + s[:120])

if __name__ == "__main__":
    main()