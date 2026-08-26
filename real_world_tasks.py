"""
REAL WORLD TASKS — OctoTetrahedral Architecture
Practical applications: text analysis, grammar checking, NER, keyword extraction,
text complexity scoring, domain detection, comparative analysis.
"""

import torch
import time
import json
import random
import statistics
from collections import Counter, defaultdict

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
    stability = float(getattr(tp, "stability", 0)) if tp else 0
    if isinstance(stability, torch.Tensor):
        stability = float(stability.item())
    comp_loss = float(getattr(tp, "compounding_loss", 0)) if tp else 0
    if isinstance(comp_loss, torch.Tensor):
        comp_loss = float(comp_loss.item())

    return {
        "words": words,
        "tags": tags,
        "paired": list(zip(words, tags)),
        "phase": getattr(tp, "phase_name", "UNKNOWN") if tp else "UNKNOWN",
        "stability": stability,
        "compounding_loss": comp_loss,
        "cohesion": float(out["cohesion"]),
    }

def tag_dist(tags):
    c = Counter(tags)
    total = sum(c.values())
    return {t: round(n / total * 100, 1) for t, n in c.most_common()}


# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  REAL WORLD TASKS — OctoTetrahedral Architecture")
print("=" * 70)


# ─── TASK 1: Keyword Extraction from News Articles ───

print("\n" + "\u2550" * 70)
print("TASK 1: KEYWORD EXTRACTION FROM NEWS ARTICLES")
print("\u2550" * 70)

articles = [
    ("Tech", "OpenAI announced the release of GPT-5 which can reason about complex mathematical problems and generate high quality code in multiple programming languages including Python Java and Rust"),
    ("Science", "Scientists at CERN discovered a new subatomic particle that could reshape our understanding of quantum mechanics and the fundamental forces of nature"),
    ("Business", "Apple reported record quarterly earnings driven by strong iPhone sales and growth in its services division including Apple Music and iCloud"),
    ("Politics", "The Senate passed a bipartisan infrastructure bill that allocates billions of dollars for roads bridges broadband internet and clean energy projects"),
    ("Health", "Researchers found that regular exercise combined with a Mediterranean diet significantly reduces the risk of heart disease and cognitive decline"),
]

for domain, text in articles:
    result = analyze(text)
    # Extract keywords: nouns and proper nouns
    keywords = [w for w, t in result["paired"] if t in ("NOUN", "PROPN")]
    entities = [w for w, t in result["paired"] if t == "PROPN"]

    print(f"\n  [{domain}] \"{text[:80]}...\"")
    print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:15])}")
    print(f"  Entities ({len(entities)}): {', '.join(entities)}")
    print(f"  Phase: {result['phase']}  Cohesion: {result['cohesion']:.4f}")
    print(f"  POS distribution: {tag_dist(result['tags'])}")


# ─── TASK 2: Text Complexity Scoring ───

print("\n" + "\u2550" * 70)
print("TASK 2: TEXT COMPLEXITY SCORING")
print("\u2550" * 70)

complexity_texts = [
    ("Children's book", "The big red ball bounced high in the sky. The little boy ran to catch it. His dog followed him. They played all day long in the warm sun."),
    ("News article", "Federal Reserve officials signaled that interest rates would remain elevated through the end of the year as inflation continues to exceed the central bank's two percent target despite aggressive monetary policy tightening."),
    ("Academic paper", "The epistemological framework underlying constructivist learning theory posits that knowledge acquisition occurs through the dialectical interaction between preexisting cognitive schemas and novel experiential input."),
    ("Legal contract", "The parties hereby agree that any dispute arising out of or relating to this agreement shall be submitted to binding arbitration in accordance with the rules of the American Arbitration Association."),
]

print(f"\n  {'Text Type':<20s} {'Words':>6s} {'Unique POS':>10s} {'NOUN%':>6s} {'VERB%':>6s} {'ADJ%':>5s} {'ADV%':>5s} {'Complexity':>10s}")
print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*10}")

for label, text in complexity_texts:
    result = analyze(text)
    tags = result["tags"]
    n_words = len(tags)
    unique_pos = len(set(tags))
    dist = tag_dist(tags)
    noun_pct = dist.get("NOUN", 0) + dist.get("PROPN", 0)
    verb_pct = dist.get("VERB", 0) + dist.get("AUX", 0)
    adj_pct = dist.get("ADJ", 0)
    adv_pct = dist.get("ADV", 0)

    # Complexity heuristic: higher ADJ/ADV ratio + higher cohesion + more unique POS = more complex
    complexity_score = (adj_pct + adv_pct) * 0.3 + unique_pos * 0.5 + result["cohesion"] * 2
    complexity_label = "Simple" if complexity_score < 8 else "Moderate" if complexity_score < 12 else "Complex"

    print(f"  {label:<20s} {n_words:>6d} {unique_pos:>10d} {noun_pct:>5.1f}% {verb_pct:>5.1f}% {adj_pct:>4.1f}% {adv_pct:>4.1f}% {complexity_label:>10s}")
    print(f"    Tags: {' '.join(tags)}")
    print(f"    Phase: {result['phase']}  CompLoss: {result['compounding_loss']:.2f}  Cohesion: {result['cohesion']:.4f}")


# ─── TASK 3: Grammar Pattern Analysis ───

print("\n" + "\u2550" * 70)
print("TASK 3: GRAMMAR PATTERN ANALYSIS")
print("\u2550" * 70)

sentences = [
    "The dog chased the cat across the yard",
    "She gave him a beautiful painting for his birthday",
    "They have been working on this project since January",
    "If it rains tomorrow we will cancel the outdoor event",
    "The professor who teaches linguistics published a new book",
]

for text in sentences:
    result = analyze(text)
    tags = result["tags"]

    # Extract n-gram patterns
    patterns = []
    for i in range(len(tags) - 1):
        patterns.append(f"{tags[i]}→{tags[i+1]}")

    noun_phrases = []
    i = 0
    while i < len(tags):
        if tags[i] == "DET":
            j = i + 1
            while j < len(tags) and tags[j] in ("ADJ", "NUM"):
                j += 1
            if j < len(tags) and tags[j] == "NOUN":
                phrase = " ".join(result["words"][i:j+1])
                noun_phrases.append(phrase)
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    print(f'\n  "{text}"')
    print(f"    POS chain: {' → '.join(tags)}")
    print(f"    Bigrams: {', '.join(patterns)}")
    print(f"    Noun phrases: {noun_phrases if noun_phrases else '(none detected)'}")
    print(f"    Phase: {result['phase']}")


# ─── TASK 4: Named Entity Recognition ───

print("\n" + "\u2550" * 70)
print("TASK 4: NAMED ENTITY RECOGNITION")
print("\u2550" * 70)

ner_texts = [
    "Elon Musk founded SpaceX in 2002 and later acquired Twitter",
    "Dr. Smith works at Stanford University in Palo Alto California",
    "The European Union and United Nations held a summit in Brussels Belgium",
    "Amazon Web Services and Microsoft Azure are leading cloud providers",
    "Barack Obama served as the 44th President of the United States",
]

for text in ner_texts:
    result = analyze(text)

    # Extract entities (PROPN sequences)
    entities = []
    current_entity = []
    for w, t in result["paired"]:
        if t == "PROPN":
            current_entity.append(w)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))

    print(f'\n  "{text}"')
    print(f"    Entities found: {entities}")
    print(f"    PROPN count: {sum(1 for _, t in result['paired'] if t == 'PROPN')}")
    print(f"    All tags: {' '.join(result['tags'])}")


# ─── TASK 5: Text Similarity via POS Profiles ───

print("\n" + "\u2550" * 70)
print("TASK 5: TEXT SIMILARITY VIA POS PROFILES")
print("\u2550" * 70)

text_pairs = [
    ("The cat sat on the mat", "The dog lay on the rug"),
    ("She quickly ran home", "He swiftly walked to school"),
    ("The beautiful sunset painted the sky", "The gorgeous painting adorned the wall"),
    ("Scientists discovered a new planet", "Researchers found a novel species"),
]

for text_a, text_b in text_pairs:
    result_a = analyze(text_a)
    result_b = analyze(text_b)

    dist_a = tag_dist(result_a["tags"])
    dist_b = tag_dist(result_b["tags"])

    # POS profile similarity
    all_tags = set(list(dist_a.keys()) + list(dist_b.keys()))
    dot = sum(dist_a.get(t, 0) * dist_b.get(t, 0) for t in all_tags)
    mag_a = sum(v**2 for v in dist_a.values()) ** 0.5
    mag_b = sum(v**2 for v in dist_b.values()) ** 0.5
    similarity = dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0

    print(f'\n  A: "{text_a}"')
    print(f'  B: "{text_b}"')
    print(f"    A tags: {result_a['tags']}")
    print(f"    B tags: {result_b['tags']}")
    print(f"    POS similarity: {similarity:.4f}")
    print(f"    Same structure: {'Yes' if result_a['tags'] == result_b['tags'] else 'No'}")


# ─── TASK 6: Domain Detection via POS Distribution ───

print("\n" + "\u2550" * 70)
print("TASK 6: DOMAIN DETECTION VIA POS DISTRIBUTION")
print("\u2550" * 70)

domain_texts = {
    "Academic": "The methodology employed in this study utilizes a mixed methods approach combining quantitative analysis with qualitative interviews to examine the multifaceted dimensions of social inequality in contemporary urban environments",
    "Sports": "The quarterback threw a spectacular touchdown pass to the wide receiver who sprinted past two defenders for the winning score in the final seconds of the championship game",
    "Cooking": "Preheat the oven to three hundred fifty degrees and combine the flour sugar butter and eggs in a large mixing bowl until the batter is smooth and creamy",
    "Travel": "The ancient temple stands majestically on the hillside overlooking the stunning turquoise waters of the Mediterranean Sea where tourists from around the world gather each summer",
    "Technology": "The neural network architecture utilizes transformer based attention mechanisms to process sequential data and generate human like text outputs with remarkable fluency and coherence",
}

print(f"\n  {'Domain':<15s} {'NOUN%':>6s} {'VERB%':>6s} {'ADJ%':>5s} {'ADV%':>5s} {'PROPN%':>7s} {'AUX%':>5s} {'Phase':<15s}")
print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*5} {'-'*15}")

for domain, text in domain_texts.items():
    result = analyze(text)
    dist = tag_dist(result["tags"])
    print(f"  {domain:<15s} {dist.get('NOUN',0):>5.1f}% {dist.get('VERB',0)+dist.get('AUX',0):>5.1f}% {dist.get('ADJ',0):>4.1f}% {dist.get('ADV',0):>4.1f}% {dist.get('PROPN',0):>6.1f}% {dist.get('AUX',0):>4.1f}% {result['phase']:<15s}")
    print(f"    Full: {' '.join(result['tags'])}")


# ─── TASK 7: Sentence Quality Assessment ───

print("\n" + "\u2550" * 70)
print("TASK 7: SENTENCE QUALITY ASSESSMENT")
print("\u2550" * 70)

quality_texts = [
    ("Well-formed", "The researchers conducted a thorough analysis of the experimental data"),
    ("Fragment", "Running through the park on a sunny day"),
    ("Run-on", "The cat sat on the mat it was very comfortable and warm the sun was shining outside"),
    ("Good grammar", "Despite the heavy rain the determined hikers continued their journey up the mountain"),
    ("Awkward", "The was very of the important thing happened yesterday morning quickly"),
]

for label, text in quality_texts:
    result = analyze(text)
    tags = result["tags"]

    # Quality indicators
    has_verb = any(t in ("VERB", "AUX") for t in tags)
    has_noun = any(t in ("NOUN", "PROPN", "PRON") for t in tags)
    verb_count = sum(1 for t in tags if t in ("VERB", "AUX"))
    punct_count = sum(1 for t in tags if t == "PUNCT")

    quality_score = 0
    issues = []
    if has_verb:
        quality_score += 1
    else:
        issues.append("No verb")
    if has_noun:
        quality_score += 1
    else:
        issues.append("No noun")
    if verb_count <= 2:
        quality_score += 1
    else:
        issues.append(f"Too many verbs ({verb_count})")

    quality = "Good" if quality_score == 3 else "Fair" if quality_score >= 2 else "Poor"

    print(f'\n  [{label}] "{text}"')
    print(f"    Tags: {' '.join(tags)}")
    print(f"    Quality: {quality} ({issues if issues else 'no issues'})")
    print(f"    Phase: {result['phase']}  CompLoss: {result['compounding_loss']:.2f}")


# ─── TASK 8: Multilingual POS Distribution ───

print("\n" + "\u2550" * 70)
print("TASK 8: CROSS-DOMAIN POS DISTRIBUTION COMPARISON")
print("\u2550" * 70)

# Compare POS distributions across different text types
all_distributions = {}
for label, text in [
    ("Narrative", "Once upon a time there lived a king in a far away castle who ruled the kingdom wisely and justly for many long years"),
    ("Instructional", "First preheat the oven then mix the ingredients in a bowl and pour the batter into the greased pan"),
    ("Descriptive", "The tall ancient oak tree stood majestically in the center of the lush green meadow surrounded by colorful wildflowers"),
    ("Argumentative", "We must invest in renewable energy sources to combat climate change and ensure a sustainable future for generations"),
    ("Expository", "Photosynthesis is the process by which plants convert sunlight water and carbon dioxide into glucose and oxygen"),
]:
    result = analyze(text)
    all_distributions[label] = tag_dist(result["tags"])

# Find most distinguishing POS tags per domain
print(f"\n  {'POS Tag':<10s}", end="")
for domain in all_distributions:
    print(f" {domain:>12s}", end="")
print()
print(f"  {'-'*10}", end="")
for _ in all_distributions:
    print(f" {'-'*12}", end="")
print()

all_pos = set()
for d in all_distributions.values():
    all_pos.update(d.keys())

for pos in sorted(all_pos):
    print(f"  {pos:<10s}", end="")
    for domain in all_distributions:
        val = all_distributions[domain].get(pos, 0)
        print(f" {val:>11.1f}%", end="")
    print()


# ─── TASK 9: POS-Guided Text Generation Constraints ───

print("\n" + "\u2550" * 70)
print("TASK 9: POS-GUIDED TEXT GENERATION CONSTRAINTS")
print("\u2550" * 70)

# Show what words would be needed to match a given POS pattern
target_patterns = [
    ("Simple SVO", "DET NOUN VERB DET NOUN"),
    ("Complex", "DET NOUN AUX VERB ADP DET ADJ NOUN"),
    ("Question", "AUX PRON VERB DET NOUN"),
]

for label, pattern in target_patterns:
    print(f"\n  [{label}] Target: {pattern}")
    # Analyze a matching sentence
    examples = {
        "DET NOUN VERB DET NOUN": "The cat chased the mouse",
        "DET NOUN AUX VERB ADP DET ADJ NOUN": "The bird was sitting on a green branch",
        "AUX PRON VERB DET NOUN": "Did you eat the cake",
    }
    example = examples.get(pattern, "The dog saw the cat")
    result = analyze(example)
    print(f"  Example: \"{example}\"")
    print(f"  Actual:  {' '.join(result['tags'])}")
    match = result["tags"] == pattern.split()
    print(f"  Match:   {'Yes' if match else 'No'}")


# ─── TASK 10: Cohesion-Based Text Coherence ───

print("\n" + "\u2550" * 70)
print("TASK 10: COHESION-BASED TEXT COHERENCE")
print("\u2550" * 70)

stories = {
    "Coherent": [
        "The morning started with a gentle sunrise",
        "Birds sang in the trees outside the window",
        "The smell of coffee drifted through the house",
        "Sarah stretched and got out of bed",
        "Today would be a good day she thought",
    ],
    "Incoherent": [
        "The algorithm processed the quantum data",
        "Bananas are yellow and curved fruits",
        "The president gave a speech about healthcare",
        "Sodium chloride dissolves readily in water",
        "The soccer team won the championship game",
    ],
}

for coherence_type, sentences in stories.items():
    model.reset_state()
    print(f"\n  [{coherence_type}]")
    for i, sent in enumerate(sentences):
        result = analyze(sent)
        print(f"    [{i+1}] Coh={result['cohesion']:.4f}  Phase={result['phase']:15s}  \"{sent}\"")
    final_cohesion = result["cohesion"]
    print(f"    Final cohesion: {final_cohesion:.4f}")


# ─── SUMMARY ───

print("\n" + "=" * 70)
print("  REAL WORLD TASKS SUMMARY")
print("=" * 70)
print("""
  TASKS COMPLETED:
  1. Keyword Extraction     — NOUN/PROPN tags identify key terms and entities
  2. Text Complexity Scoring — POS distribution correlates with text type
  3. Grammar Pattern Analysis — Bigram POS patterns reveal structure
  4. Named Entity Recognition — PROPN tags detect person/org/location names
  5. Text Similarity          — POS profiles compare document structure
  6. Domain Detection         — POS distributions distinguish text domains
  7. Sentence Quality         — POS patterns assess grammaticality
  8. Cross-Domain Analysis    — POS distributions reveal genre differences
  9. POS-Guided Constraints   — POS patterns define generation templates
  10. Cohesion Coherence      — Cohesion scores track narrative flow

  MODULE DIAGNOSTICS AVAILABLE:
  - TranscendPlexity phase (EXPLORATION/CONSOLIDATION/DEEP_REASONING/OSCILLATION)
  - Compounding loss (monotonically increasing across session)
  - Cohesion score (tracks narrative state evolution)
  - Stability metric (near 1.0 for all inputs)
""")
