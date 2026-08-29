#!/usr/bin/env python3
"""Retrieval + ranking chat demo for OctoTetrahedral v8.

Answers are NOT generated; they are retrieved verbatim from a local corpus and
ranked by the LM's naturalness score (answer-span perplexity conditioned on
"Question : <q> Response : <answer>"). The model's measured strength is scoring,
so this uses it correctly instead of asking it to hallucinate open-ended text.
"""
import argparse
import difflib
import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module=".*transcendplexity_integration")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_transformer import OctoTransformerLM, CHAR_PAD, BOS_ID, EOS_ID

STOPWORDS = frozenset("""
a about after all also am an and any are as at be because been before being both but by
can could did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just like more
most my myself no nor not now of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves
""".split())

WORD_RE = re.compile(r"[a-zA-Z]+|\d+")


class ChatBot:
    def __init__(self, model, word_vocab, char_vocab, corpus_path, device,
                 topk=12, rerank_top=6, max_answer=42, online=True):
        self.model = model
        self.wv = word_vocab
        self.cv = char_vocab
        self.device = device
        self.topk = topk
        self.rerank_top = rerank_top
        self.max_answer = max_answer
        self.online = online
        self.max_len = getattr(model, "max_len", 128)
        self.sentences, self.index, self.df, self.n_docs, self.vocab_tokens = self._build_index(corpus_path)

    def _build_index(self, corpus_path):
        sentences = []
        with open(corpus_path) as f:
            for line in f:
                s = json.loads(line)["text"].strip()
                words = s.split()
                if 3 <= len(words) <= 60:
                    sentences.append(words)
        index = defaultdict(list)
        df = defaultdict(int)
        for i, words in enumerate(sentences):
            seen = set()
            tf = defaultdict(int)
            for w in words:
                for tok in WORD_RE.findall(w.lower()):
                    if len(tok) >= 2 and tok not in STOPWORDS:
                        tf[tok] += 1
                        seen.add(tok)
            for tok, c in tf.items():
                index[tok].append((i, c))
                df[tok] += 1
        return sentences, index, df, len(sentences), set(index.keys())

    def _suggest(self, missing):
        out = {}
        for tok in missing:
            if self.df.get(tok, 0) > 0:
                continue
            cands = [t for t in difflib.get_close_matches(
                        tok, self.vocab_tokens, n=6, cutoff=0.7)
                     if self.df.get(t, 0) >= 2]
            if cands:
                out[tok] = cands[:2]
        return out

    def _web_search(self, question):
        import urllib.parse
        import urllib.request
        HDRS = {"User-Agent": "OctoTetrahedralRAG/1.0 (local research demo)"}
        cands = []

        def add(src, text):
            for part in re.split(r"(?<=[.!?])\s+", text):
                w = part.split()
                if 4 <= len(w) <= 50:
                    cands.append((src, w))

        def get_json(url):
            return json.load(urllib.request.urlopen(
                urllib.request.Request(url, headers=HDRS), timeout=8))

        try:
            q = urllib.parse.quote(question)
            u = (f"https://api.duckduckgo.com/?q={q}"
                 "&format=json&no_html=1&skip_disambig=1")
            d = get_json(u)
            if d.get("AbstractText"):
                add("web (duckduckgo)", d["AbstractText"])
            for t in d.get("RelatedTopics", []) or []:
                if not isinstance(t, dict):
                    continue
                for sub in t.get("Topics", []) or []:
                    if sub.get("Text"):
                        add("web (duckduckgo)", sub["Text"])
                if t.get("Text"):
                    add("web (duckduckgo)", t["Text"])
        except Exception:
            pass

        try:
            q = urllib.parse.quote(question)
            u = (f"https://en.wikipedia.org/w/api.php?action=query"
                 f"&format=json&list=search&srsearch={q}&srlimit=4")
            d = get_json(u)
            titles = [h["title"] for h in d.get("query", {}).get("search", [])]
            for title in titles[:2]:
                t = urllib.parse.quote(title)
                u2 = (f"https://en.wikipedia.org/w/api.php?action=query"
                      f"&format=json&prop=extracts&exintro&explaintext&titles={t}")
                d2 = get_json(u2)
                for p in d2.get("query", {}).get("pages", {}).values():
                    if p.get("extract"):
                        add(f"web: wikipedia: {p['title']}", p["extract"])
        except Exception:
            pass

        return cands[:12]

    def _retrieve(self, query_words, n):
        scores = defaultdict(float)
        idf_denom = self.n_docs
        for tok in query_words:
            idf = math.log2(1 + idf_denom / max(1, self.df.get(tok, 0)))
            for sid, tf in self.index.get(tok, ()):
                scores[sid] += (1 + math.log2(1 + tf)) * idf
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [sid for sid, _ in ranked]

    def _encode(self, words):
        ids = [BOS_ID] + [self.wv.get(w, 1) for w in words] + [EOS_ID]
        ids = ids[: self.max_len]
        raw = ["<BOS>"] + words + ["<EOS>"]
        raw = raw[: self.max_len]
        char_ids = torch.zeros(len(ids), 30, dtype=torch.long)
        for j, w in enumerate(raw):
            cs = [self.cv.get(c, 1) for c in w.lower()[:30]]
            while len(cs) < 30:
                cs.append(CHAR_PAD)
            char_ids[j] = torch.tensor(cs[:30])
        return torch.tensor([ids]), char_ids.unsqueeze(0)

    def _answer_span_ppl(self, prefix, answer):
        words = prefix + answer
        full = words[: self.max_len - 1]
        wids, cids = self._encode(full)
        wids, cids = wids.to(self.device), cids.to(self.device)
        with torch.no_grad():
            out = self.model(wids, cids, targets=wids)
        logits = out["lm_logits"][:, :-1, :]
        targets = wids[:, 1:]
        ans_start = len(prefix)
        lo, hi = ans_start, min(len(full) - 1, ans_start + len(answer))
        if hi <= lo or hi < 0 or lo >= logits.size(1):
            return float("inf")
        shift = logits[0, lo:hi, :]
        tgt = targets[0, lo:hi]
        loss = F.cross_entropy(shift, tgt, reduction="mean").item()
        return math.exp(min(loss, 30))

    def _tokens(self, words):
        toks = []
        for w in words:
            for tok in WORD_RE.findall(w.lower()):
                if len(tok) >= 2 and tok not in STOPWORDS:
                    toks.append(tok)
        return toks

    def _coverage(self, cand_tokens, q_toks):
        idf_sum = 0.0
        matched = 0.0
        ct = set(cand_tokens)
        for tok in q_toks:
            idf = math.log2(1 + self.n_docs / max(1, self.df.get(tok, 0)))
            idf_sum += idf
            if tok in ct:
                matched += idf
        return matched / idf_sum if idf_sum > 0 else 0.0

    def ask(self, question):
        q_words = question.split()
        q_toks = self._tokens(q_words)
        missing = [t for t in q_toks if self.df.get(t, 0) == 0]
        maybe = self._suggest(missing) if missing else {}
        if not q_toks:
            return {"answers": [], "question": question}
        cands = self._retrieve(q_toks, self.topk)
        reason = ("no corpus hits" if not cands
                  else f"no topically relevant corpus content for: {' '.join(q_toks)}")

        prefix = ["Question", ":"] + q_words + ["Response", ":"]
        scored = []
        for sid in cands[: self.rerank_top * 2]:
            answer = self.sentences[sid]
            if len(answer) > self.max_answer:
                answer = answer[: self.max_answer]
            cov = self._coverage(self._tokens(answer), q_toks)
            if cov < 0.6:
                continue
            ppl = self._answer_span_ppl(prefix, answer)
            scored.append({"ppl": ppl, "sid": sid,
                           "text": " ".join(answer), "coverage": cov,
                           "source": "local"})
        if not scored and self.online:
            for src, words in self._web_search(question):
                words = words[: self.max_answer]
                ppl = self._answer_span_ppl(prefix, words)
                if ppl < 4.0:
                    scored.append({"ppl": ppl, "sid": -1,
                                   "text": " ".join(words),
                                   "coverage": 1.0, "source": src})
            scored.sort(key=lambda x: x["ppl"])
        if not scored:
            return {"answers": [], "question": question,
                    "reason": reason,
                    "maybe": maybe}

        best = scored[0]
        median = sorted(x["ppl"] for x in scored)[len(scored) // 2] if scored else float("inf")
        beat_peers = best["ppl"] < 0.87 * median if median != float("inf") else False
        confidence = "HIGH" if best["ppl"] < 2.0 and beat_peers else (
            "MEDIUM" if best["ppl"] < 3.0 else "LOW")
        return {
            "question": question,
            "answers": scored,
            "best": best["text"],
            "best_ppl": best["ppl"],
            "confidence": confidence,
            "maybe": maybe,
            "source": best.get("source", "local"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/octo_transformer_best.pt")
    ap.add_argument("--corpus", default="data/transcripts.jsonl")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--rerank-top", type=int, default=6)
    ap.add_argument("--question", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-online", action="store_true",
                    help="Disable web search fallback (DuckDuckGo + Wikipedia)")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wc, cc = ckpt["word_vocab"], ckpt["char_vocab"]
    model = OctoTransformerLM(len(wc), len(cc), **ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
          f"(eval_ppl {ckpt.get('eval_ppl', '?'):.2f}) | device={device}")

    bot = ChatBot(model, wc, cc, args.corpus, device,
                  topk=args.topk, rerank_top=args.rerank_top,
                  online=not args.no_online)
    print(f"Corpus: {args.corpus} ({bot.n_docs:,} sentences)")

    def render_suggestions(res):
        for tok, suggs in (res.get("maybe") or {}).items():
            print(f"      Did you mean \"{suggs[0]}\" for '{tok}'? (corpus suggestion)")
            if len(suggs) > 1:
                print(f"      or \"{suggs[1]}\"?")

    def render(res):
        if res.get("answers") is None:
            print(f"\n  (nothing searchable in: {res['question']})")
            return
        if not res["answers"]:
            reason = res.get("reason", "no corpus hits")
            print(f"\n  [{reason}] ({res['question']})")
            render_suggestions(res)
            return
        src = res.get("source", "local")
        print(f"\n  [{res['confidence']} · {src}] {res['best']}")
        for a in res["answers"][1:4]:
            print(f"      alt ({a['ppl']:.2f} · {a.get('source','local')}): {a['text']}")
        render_suggestions(res)

    if args.question:
        t0 = time.time()
        res = bot.ask(args.question)
        print(f"\nQ: {args.question}")
        render(res)
        print(f"\n({time.time()-t0:.1f}s)")
        return

    print("Chat demo (retrieved verbatim + ranked by LM naturalness score; "
          "answer text is NOT generated). Falls back to web search "
          "(DuckDuckGo + Wikipedia) when the corpus has no answer. 'quit' to exit.")
    while True:
        try:
            q = input("\nQ> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        res = bot.ask(q)
        render(res)


if __name__ == "__main__":
    main()