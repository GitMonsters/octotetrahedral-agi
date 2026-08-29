import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import gen_instructions as gi

random.seed(11)

OUT = "data/instructions_v2.jsonl"
TRANSCRIPTS = "data/transcripts.jsonl"
MAX_RESP_WORDS = 20

STOP = set("""a an and are as at be but by for from had has have he her his i if in into
is it its of on or she that the their them then there these they this to was we were
what when where which who will with you your not so do does did can could would should
about over after before same too very also just than his her them those only own same
two other new more most such no nor one our out up down off what how why where""".split())

TEMPLATES = [
    "What is {k} ?",
    "What is known about {k} ?",
    "Tell me about {k} .",
    "What does {k} mean ?",
    "How does {k} work ?",
    "Why is {k} important ?",
    "Give me an overview of {k} .",
    "Show me about {k} .",
    "What can {k} tell us ?",
]

def keyword(sentence):
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", sentence.lower())
    freq = {}
    for w in words:
        if len(w) >= 5 and w not in STOP and not w.endswith(("ly", "ing", "tion")):
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return None
    return max(sorted(freq), key=lambda w: (freq[w], len(w)))

def gen():
    import torch
    wc = torch.load("checkpoints/octo_transformer_best.pt",
                    map_location="cpu", weights_only=False)["word_vocab"]
    samples = []
    used = set()
    with open(TRANSCRIPTS) as f:
        sents = [json.loads(l)["text"] for l in f]
    for s in sents:
        k = keyword(s)
        if not k or k not in wc:
            continue
        inst = random.choice(TEMPLATES).format(k=k)
        resp = " ".join(s.split()[:MAX_RESP_WORDS])
        text = f"Question : {inst}\nResponse : {resp}"
        if text in used:
            continue
        used.add(text)
        samples.append({"text": text})
    for user, ai in gi.CHAT_PAIRS:
        text = f"Question : {user}\nResponse : {ai}"
        if text not in used:
            used.add(text)
            samples.append({"text": text})
    random.shuffle(samples)
    return samples

if __name__ == "__main__":
    samples = gen()
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} samples -> {OUT}")