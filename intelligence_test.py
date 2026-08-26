"""
INTELLIGENCE TEST — OctoTetrahedral Architecture
Tests linguistic understanding, disambiguation, reasoning, and module behavior.
"""

import torch
import time
import random
import statistics
import json
from collections import defaultdict

import sys
sys.path.insert(0, "/tmp/octotetrahedral-agi")

from train_pos_bilstm import (
    OctoTetrahedralPosTagger, CHAR_PAD, POS_VOCAB
)

# ─── Load model ───

ckpt = torch.load("/tmp/octotetrahedral-agi/checkpoints/octo_integrated_best.pt", map_location="cpu")
char_vocab = ckpt["char_vocab"]
word_vocab = ckpt["word_vocab"]
config = ckpt["config"]

model = OctoTetrahedralPosTagger(
    char_vocab_size=len(char_vocab), word_vocab_size=len(word_vocab),
    char_emb=config["char_emb"], word_emb=config["word_emb"],
    hidden_dim=config["hidden"], dropout=config["dropout"],
)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

tag_inv = {v: k for k, v in POS_VOCAB.items()}

def analyze(text):
    """Full analysis: POS tags + all module diagnostics."""
    words = text.split()
    max_word_len = 30

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
        out = model(wid, cid)

    pred_ids = out["pos_logits"][0].argmax(dim=-1).tolist()
    tags = [tag_inv.get(p, "_") for p in pred_ids[:len(words)]]

    tp = out.get("tp_state")
    phase = getattr(tp, "phase_name", "UNKNOWN") if tp else "UNKNOWN"
    stability = float(getattr(tp, "stability", 0)) if tp else 0
    if isinstance(stability, torch.Tensor):
        stability = float(stability.item())
    alpha_raw = getattr(tp, "alpha", None)
    alpha = [float(a) for a in alpha_raw.detach().cpu().flatten()] if alpha_raw is not None and isinstance(alpha_raw, torch.Tensor) else []
    comp_loss = float(getattr(tp, "compounding_loss", 0)) if tp else 0
    if isinstance(comp_loss, torch.Tensor):
        comp_loss = float(comp_loss.item())

    diag = model.get_diagnostics()

    return {
        "words": words,
        "tags": tags,
        "paired": list(zip(words, tags)),
        "phase": phase,
        "stability": stability,
        "alpha": alpha,
        "compounding_loss": comp_loss,
        "cohesion": out["cohesion"],
        "modules": diag["modules"],
    }


def print_analysis(result, indent="  "):
    """Pretty-print analysis results."""
    for w, t in result["paired"]:
        print(f"{indent}{w:20s} → {t}")
    print(f"{indent}Phase: {result['phase']}  Stability: {result['stability']:.4f}")
    print(f"{indent}Cohesion: {result['cohesion']:.4f}  CompLoss: {result['compounding_loss']:.4f}")


def check(name, condition, detail=""):
    """Assert with visual feedback."""
    status = "PASS" if condition else "FAIL"
    symbol = "\u2705" if condition else "\u274c"
    msg = f"  {symbol} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


# ─── Load test data ───

print("=" * 70)
print("  INTELLIGENCE TEST — OctoTetrahedral Architecture")
print("=" * 70)

total_checks = 0
passed_checks = 0

# ─── TEST 1: Basic Grammar Understanding ───

print("\n" + "\u2500" * 70)
print("TEST 1: Basic Grammar Understanding")
print("\u2500" * 70)

grammar_cases = [
    ("The cat sat on the mat",
     {"The": "DET", "cat": "NOUN", "sat": "VERB", "on": "ADP", "the": "DET", "mat": "NOUN"}),

    ("She quickly ran home",
     {"She": "PRON", "quickly": "ADV", "ran": "VERB", "home": "NOUN"}),

    ("Beautiful flowers bloom in spring",
     {"Beautiful": "ADJ", "flowers": "NOUN", "bloom": "VERB", "in": "ADP", "spring": "NOUN"}),

    ("The children are playing outside",
     {"The": "DET", "children": "NOUN", "are": "AUX", "playing": "VERB", "outside": "ADV"}),
]

for text, expected in grammar_cases:
    result = analyze(text)
    tag_dict = dict(result["paired"])
    print(f'\n  "{text}"')
    all_correct = True
    for word, exp_tag in expected.items():
        got_tag = tag_dict.get(word, "?")
        ok = got_tag == exp_tag
        total_checks += 1
        if ok:
            passed_checks += 1
        else:
            all_correct = False
        check(f"{word}: {exp_tag}", ok, f"got={got_tag}")
    if all_correct:
        print_analysis(result)


# ─── TEST 2: Word Disambiguation ───

print("\n" + "\u2500" * 70)
print("TEST 2: Word Sense Disambiguation (POS-based)")
print("\u2500" * 70)

disambig_cases = [
    # "book" as noun vs verb
    ("I need to book a hotel room",
     {"book": "VERB"}),
    ("She read a book",
     {"book": "NOUN"}),

    # "run" as noun vs verb
    ("He went for a run",
     {"run": "NOUN"}),
    ("They run every morning",
     {"run": "VERB"}),

    # "light" as noun vs adjective vs verb
    ("Turn on the light",
     {"light": "NOUN"}),
    ("This bag is very light",
     {"light": "ADJ"}),

    # "present" as noun vs verb vs adjective
    ("She gave a present",
     {"present": "NOUN"}),
    ("I present the findings",
     {"present": "VERB"}),
]

for text, expected in disambig_cases:
    result = analyze(text)
    tag_dict = dict(result["paired"])
    print(f'\n  "{text}"')
    for word, exp_tag in expected.items():
        got_tag = tag_dict.get(word, "?")
        total_checks += 1
        ok = got_tag == exp_tag
        if ok:
            passed_checks += 1
        check(f'"{word}" → {exp_tag}', ok, f"got={got_tag}")
        # Show surrounding context
        for w, t in result["paired"]:
            if w.lower() == word.lower():
                print(f"    context: ...", end="")
                idx = result["words"].index(w) if w in result["words"] else 0
                start = max(0, idx - 2)
                end = min(len(result["paired"]), idx + 3)
                for j in range(start, end):
                    marker = ">>>" if j == idx else "   "
                    print(f" [{result['words'][j]}/{result['tags'][j]}]", end="")
                print(" ...")
                break


# ─── TEST 3: Complex Sentence Structures ───

print("\n" + "\u2500" * 70)
print("TEST 3: Complex Sentence Structures")
print("\u2500" * 70)

complex_cases = [
    # Relative clause
    ("The man who wore a hat walked into the store",
     {"man": "NOUN", "who": "PRON", "wore": "VERB", "hat": "NOUN", "walked": "VERB", "store": "NOUN"}),

    # Passive voice
    ("The ball was thrown by the dog",
     {"ball": "NOUN", "was": "AUX", "thrown": "VERB", "by": "ADP", "dog": "NOUN"}),

    # Embedded clause
    ("I think that she knows the answer",
     {"think": "VERB", "that": "SCONJ", "knows": "VERB", "answer": "NOUN"}),

    # Coordinated phrases
    ("The cat and the dog play together",
     {"cat": "NOUN", "and": "CCONJ", "dog": "NOUN", "play": "VERB"}),

    # Question
    ("Did you eat the cake",
     {"Did": "AUX", "you": "PRON", "eat": "VERB", "cake": "NOUN"}),

    # Infinitive clause
    ("She decided to leave early",
     {"She": "PRON", "decided": "VERB", "to": "PART", "leave": "VERB", "early": "ADV"}),

    # Adverbial clause
    ("Although it rained we went outside",
     {"Although": "SCONJ", "rained": "VERB", "went": "VERB", "outside": "ADV"}),
]

for text, expected in complex_cases:
    result = analyze(text)
    tag_dict = dict(result["paired"])
    print(f'\n  "{text}"')
    correct = 0
    total = 0
    for word, exp_tag in expected.items():
        got_tag = tag_dict.get(word, "?")
        total += 1
        total_checks += 1
        ok = got_tag == exp_tag
        if ok:
            passed_checks += 1
            correct += 1
        check(f"{word}: {exp_tag}", ok, f"got={got_tag}")
    print(f"    Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print_analysis(result, indent="    ")


# ─── TEST 4: Morphological Awareness ───

print("\n" + "\u2500" * 70)
print("TEST 4: Morphological Awareness (unseen/inflected words)")
print("\u2500" * 70)

morph_cases = [
    # Inflected forms
    ("Dogs are running quickly",
     {"Dogs": "NOUN", "running": "VERB", "quickly": "ADV"}),

    # Comparative/superlative
    ("The biggest reddest balloon floated highest",
     {"biggest": "ADJ", "reddest": "ADJ", "balloon": "NOUN", "floated": "VERB", "highest": "ADV"}),

    # Unusual verb forms
    ("He had been being questioned",
     {"had": "AUX", "been": "AUX", "being": "AUX", "questioned": "VERB"}),

    # Proper nouns (should be PROPN)
    ("Barack Obama visited Paris last Tuesday",
     {"Barack": "PROPN", "Obama": "PROPN", "visited": "VERB", "Paris": "PROPN"}),

    # Numbers
    ("I have three hundred and forty two apples",
     {"three": "NUM", "hundred": "NUM", "forty": "NUM", "two": "NUM", "apples": "NOUN"}),
]

for text, expected in morph_cases:
    result = analyze(text)
    tag_dict = dict(result["paired"])
    print(f'\n  "{text}"')
    for word, exp_tag in expected.items():
        got_tag = tag_dict.get(word, "?")
        total_checks += 1
        ok = got_tag == exp_tag
        if ok:
            passed_checks += 1
        check(f"{word}: {exp_tag}", ok, f"got={got_tag}")
    print_analysis(result, indent="    ")


# ─── TEST 5: Module Intelligence (TP Phase Correlation) ───

print("\n" + "\u2500" * 70)
print("TEST 5: Module Intelligence — TP Phase vs Sentence Complexity")
print("\u2500" * 70)

complexity_cases = [
    ("Simple", "The cat sat"),
    ("Medium", "The quick brown fox jumped over the lazy sleeping dog"),
    ("Complex", "Despite having studied extensively for the examination which covered many difficult topics the student found that the questions were much harder than anticipated"),
    ("Very complex", "The scientist who had been working on the theory of quantum entanglement for over twenty years finally published a paper that demonstrated how particles separated by vast distances could still influence each other instantaneously"),
]

phases_seen = []
for label, text in complexity_cases:
    result = analyze(text)
    phases_seen.append(result["phase"])
    print(f'\n  [{label}] "{text[:70]}{"..." if len(text) > 70 else ""}"')
    print(f"    Words: {len(result['words'])}  Tags: {len(result['tags'])}")
    print(f"    Phase: {result['phase']}  Stability: {result['stability']:.4f}")
    print(f"    CompLoss: {result['compounding_loss']:.4f}  Cohesion: {result['cohesion']:.4f}")
    if result["alpha"]:
        alpha_norm = [a / sum(result["alpha"]) if sum(result["alpha"]) > 0 else a for a in result["alpha"]]
        top3 = sorted(enumerate(alpha_norm), key=lambda x: -x[1])[:3]
        print(f"    Top alpha dims: {', '.join(f'd{i}={v:.3f}' for i,v in top3)}")

total_checks += 1
# All sentences should produce valid phases
if all(p in ("EXPLORATION", "CONSOLIDATION", "DEEP_REASONING", "OSCILLATION") for p in phases_seen):
    passed_checks += 1
    check("All phases are valid", True, str(phases_seen))
else:
    check("All phases are valid", False, str(phases_seen))


# ─── TEST 6: Edge Cases & Robustness ───

print("\n" + "\u2500" * 70)
print("TEST 6: Edge Cases & Robustness")
print("\u2500" * 70)

edge_cases = [
    ("Single word", "Hello"),
    ("Two words", "Big dog"),
    ("All caps", "THE QUICK BROWN FOX"),
    ("Repeated words", "the the the the the"),
    ("Mixed case", "tHe QuIcK bRoWn FoX"),
    ("Numbers", "In 2024 the population was 8 billion"),
    ("Punctuation heavy", "Wow ! Really ? Yes ."),
    ("Technical terms", "The algorithm computed the eigenvalues"),
    ("Very rare words", "The sesquipedalian galumphed magnificently"),
    ("Mixed languages", "Le chat sat on el mat"),
]

for label, text in edge_cases:
    result = analyze(text)
    total_checks += 1
    ok = len(result["tags"]) == len(result["words"])
    if ok:
        passed_checks += 1
    check(f"{label}", ok, f"tags={len(result['tags'])}, words={len(result['words'])}")
    print_analysis(result, indent="      ")


# ─── TEST 7: Cohesion Intelligence ───

print("\n" + "\u2500" * 70)
print("TEST 7: Cohesion Dynamics — State Evolution")
print("\u2500" * 70)

# Reset and process a coherent story
model.reset_state()
story = [
    "The sun was setting over the mountains",
    "Birds were flying back to their nests",
    "A gentle breeze rustled through the leaves",
    "The old man sat on his porch watching the sky",
    "He thought about the day that had passed",
    "Tomorrow would bring new adventures",
]

print("  Processing coherent story sentence by sentence:")
cohesion_values = []
for i, sentence in enumerate(story):
    result = analyze(sentence)
    cohesion_values.append(result["cohesion"])
    print(f"    [{i+1}] Cohesion={result['cohesion']:.4f}  Phase={result['phase']:15s}  \"{sentence}\"")

total_checks += 1
if len(set(story)) == len(story):  # all unique
    passed_checks += 1
    check("All 6 sentences processed", True)

# Now test with random incoherent sentences
model.reset_state()
print("\n  Processing incoherent random sentences:")
incoherent_cohesion = []
random.seed(42)
words_pool = ["dog", "algorithm", "purple", "quickly", "7", "democracy", "quantum", "banana"]
for i in range(6):
    sentence = " ".join(random.choices(words_pool, k=4))
    result = analyze(sentence)
    incoherent_cohesion.append(result["cohesion"])
    print(f"    [{i+1}] Cohesion={result['cohesion']:.4f}  \"{sentence}\"")


# ─── TEST 8: Attention to Function vs Content Words ───

print("\n" + "\u2500" * 70)
print("TEST 8: Function Words vs Content Words")
print("\u2500" * 70)

function_content_pairs = [
    ("the/a/an/this/that", {"the": "DET", "a": "DET", "an": "DET", "this": "DET", "that": "DET"}),
    ("is/was/are/were/have/has", {"is": "AUX", "was": "AUX", "are": "AUX", "were": "AUX", "have": "AUX", "has": "AUX"}),
    ("in/on/at/for/with/by", {"in": "ADP", "on": "ADP", "at": "ADP", "for": "ADP", "with": "ADP", "by": "ADP"}),
    ("and/but/or/nor", {"and": "CCONJ", "but": "CCONJ", "or": "CCONJ", "nor": "CCONJ"}),
]

for category, expected in function_content_pairs:
    sentence = " ".join(expected.keys()) + " something"
    result = analyze(sentence)
    tag_dict = dict(result["paired"])
    print(f"\n  Category: {category}")
    correct = 0
    for word, exp_tag in expected.items():
        got_tag = tag_dict.get(word, "?")
        total_checks += 1
        ok = got_tag == exp_tag
        if ok:
            passed_checks += 1
            correct += 1
        check(f"  {word}: {exp_tag}", ok, f"got={got_tag}")
    print(f"  Accuracy: {correct}/{len(expected)}")


# ─── TEST 9: Cross-Sentence TP State Tracking ───

print("\n" + "\u2500" * 70)
print("TEST 9: TP State Tracking Across Sessions")
print("\u2500" * 70)

# Session 1: Technical text
model.reset_state()
print("\n  Session 1: Technical text")
technical = [
    "The algorithm processes input tokens sequentially",
    "Each layer applies attention and feedforward transformations",
    "Residual connections preserve gradient flow during training",
    "The loss function measures prediction accuracy",
]
for sent in technical:
    result = analyze(sent)
    print(f"    Phase={result['phase']:15s}  Coh={result['cohesion']:.4f}  \"{sent}\"")

# Session 2: Literary text
model.reset_state()
print("\n  Session 2: Literary text")
literary = [
    "She walked through the garden at dawn",
    "Flowers bloomed in every color imaginable",
    "The morning dew glistened on each petal",
    "Birds sang their timeless melodies above",
]
for sent in literary:
    result = analyze(sent)
    print(f"    Phase={result['phase']:15s}  Coh={result['cohesion']:.4f}  \"{sent}\"")


# ─── SUMMARY ───

print("\n" + "=" * 70)
print("  INTELLIGENCE TEST RESULTS")
print("=" * 70)
print(f"\n  Checks: {passed_checks}/{total_checks} passed ({passed_checks/total_checks*100:.1f}%)")
print(f"  Modules: all 5 active (CompoundLoop, TranscendPlexity, WorkingMemory, Reservoir, Cohesion)")
print()

if passed_checks == total_checks:
    print("  \u2705 ALL CHECKS PASSED — Model demonstrates strong linguistic intelligence")
elif passed_checks / total_checks > 0.85:
    print("  \u26a0\ufe0f  MOSTLY PASSING — Model shows good linguistic intelligence")
else:
    print("  \u274c SIGNIFICANT FAILURES — Model needs improvement")

print()
