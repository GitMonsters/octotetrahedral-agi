import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gen_instructions as gi

random.seed(7)

OUT = "data/instructions.jsonl"
TRANSCRIPTS = "data/transcripts.jsonl"
MAX_RESP_WORDS = 24

STOP = set("""a an and are as at be but by for from had has have he her his i if in into
is it its of on or she that the their them then there these they this to was we were
what when where which who will with you your not so do does did can could would should
about over after before same too very also just than his her them those only own same
two other new more most such no nor one our out up down off""".split())

TEMPLATES = [
    "What is {k}?",
    "Explain {k}.",
    "Tell me about {k}.",
    "What should I know about {k}?",
    "What are the main ideas behind {k}?",
    "Why do people study {k}?",
    "What makes {k} interesting?",
    "How is {k} used in practice?",
    "Give me an overview of {k}.",
]

def keyword(sentence):
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", sentence.lower())
    freq = {}
    for w in words:
        if len(w) >= 5 and w not in STOP:
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return None
    best = max(sorted(freq), key=lambda w: (freq[w], len(w)))
    return best if not best.endswith(("ly", "ing")) and freq[best] >= 1 else None

def gen():
    samples = []
    with open(TRANSCRIPTS) as f:
        sents = [json.loads(l)["text"] for l in f]
    n_used = 0
    for s in sents:
        k = keyword(s)
        if not k:
            continue
        words = s.split()
        resp = " ".join(words[:MAX_RESP_WORDS])
        inst = random.choice(TEMPLATES).format(k=k)
        samples.append({"text": f"Instruction: {inst}\nResponse: {resp}"})
        n_used += 1
    for user, ai in gi.CHAT_PAIRS:
        samples.append({"text": f"Instruction: {user}\nResponse: {ai}"})
    random.shuffle(samples)
    return samples

if __name__ == "__main__":
    samples = gen()
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    word_counts = sum(len(s["text"].split()) for s in samples) / len(samples)
    print(f"wrote {len(samples)} instruction samples -> {OUT} (avg {word_counts:.0f} words)")