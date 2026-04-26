//! domain_analysis.rs — Input domain classification for OctoTetrahedral
//!
//! Classifies a free-text prompt into active cognitive domains, then maps each
//! domain to the limbs expected to activate (high confidence) or stay quiet
//! (low confidence).  The actual limb confidences from the fan-out step are
//! compared against these expectations to produce a `domain_sensitivity` report
//! included in every `/infer` response.
//!
//! Domain taxonomy
//! ───────────────
//!  ethics     — risk/safety/harm scenarios (Friday deploys, medical emergencies, etc.)
//!  empathy    — theory-of-mind / other-agent states
//!  emotion    — self-referential affect (fear, grief, anxiety)
//!  planning   — goal-directed sequencing; LOW until a concrete goal is detected
//!  reasoning  — logical/analytical queries
//!  language   — language/translation/summarisation tasks
//!  memory     — recall / reference to past context
//!  spatial    — spatial/geometric/visual layout queries

use std::collections::HashMap;

use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────────────
// Trigger tables
// ─────────────────────────────────────────────────────────────────────────────

/// (domain, keywords that activate it)
const DOMAIN_TRIGGERS: &[(&str, &[&str])] = &[
    (
        "ethics",
        &[
            "risk", "deploy", "friday", "production", "outage", "rollout",
            "critical", "dangerous", "harm", "hurt", "unsafe", "safety",
            "chest pain", "emergency", "urgent", "life-threatening",
            "breach", "vulnerability", "exploit", "privacy", "gdpr",
            "bias", "discriminat", "unethical",
        ],
    ),
    (
        "empathy",
        &[
            "they feel", "she feels", "he feels", "feels like", "how are you",
            "understand them", "perspective", "their point of view", "empathize",
            "put yourself in", "imagine being", "what would they",
            "affected people", "community impact", "stakeholders",
        ],
    ),
    (
        "emotion",
        &[
            "i feel", "i'm scared", "i'm worried", "i'm sad", "anxious",
            "depressed", "grieving", "suffering", "overwhelmed", "lonely",
            "hopeless", "frustrated", "angry", "afraid", "nervous",
            "excited", "happy", "joy",
        ],
    ),
    (
        "planning",
        &[
            "plan", "steps", "how to", "schedule", "roadmap", "milestone",
            "deadline", "timeline", "phases", "strategy", "next steps",
            "action item", "todo", "checklist", "sprint", "backlog",
        ],
    ),
    (
        "reasoning",
        &[
            "why", "because", "therefore", "if.*then", "prove", "infer",
            "deduce", "logic", "argument", "hypothesis", "evidence",
            "cause", "effect", "conclusion",
        ],
    ),
    (
        "language",
        &[
            "translate", "summarize", "summarise", "rephrase", "paraphrase",
            "grammar", "spell", "write", "draft", "compose", "edit",
            "proofread", "code review",
        ],
    ),
    (
        "memory",
        &[
            "remember", "recall", "earlier", "previously", "last time",
            "you said", "as we discussed", "history", "context",
        ],
    ),
    (
        "spatial",
        &[
            "layout", "position", "coordinate", "geometry", "shape",
            "distance", "map", "grid", "rotate", "flip", "arrange",
            "visualize", "diagram", "chart", "graph",
        ],
    ),
];

/// Domains where HIGH confidence is expected when triggered
const DOMAIN_HIGH_LIMBS: &[(&str, &[&str])] = &[
    ("ethics",    &["ethics"]),
    ("empathy",   &["empathy", "emotion"]),
    ("emotion",   &["emotion", "empathy"]),
    ("planning",  &["planning"]),
    ("reasoning", &["reasoning", "metacognition"]),
    ("language",  &["language"]),
    ("memory",    &["memory"]),
    ("spatial",   &["spatial", "visualization"]),
];

/// Planning stays LOW when no concrete goal keywords are detected
const PLANNING_GOAL_KEYWORDS: &[&str] = &[
    "plan", "steps", "how to", "roadmap", "milestone",
    "deadline", "timeline", "strategy", "checklist",
];

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Debug)]
pub struct DomainHit {
    /// Domain name
    pub domain: String,
    /// Keywords from the input that triggered this domain
    pub triggers: Vec<String>,
    /// Limbs expected to show HIGH activation for this domain
    pub expected_high: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct LimbExpectation {
    pub limb: String,
    pub expected: &'static str,   // "high" | "low" | "neutral"
    pub actual_confidence: f32,
    pub met: bool,
}

#[derive(Serialize, Debug)]
pub struct DomainAnalysis {
    /// Active domains detected in the prompt
    pub active_domains: Vec<DomainHit>,
    /// Whether planning has a concrete goal (false → expected LOW)
    pub planning_has_goal: bool,
    /// Per-limb expectation vs actual
    pub limb_expectations: Vec<LimbExpectation>,
    /// Overall sensitivity score: fraction of expectations met (0.0–1.0)
    pub sensitivity_score: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis logic
// ─────────────────────────────────────────────────────────────────────────────

/// Classify `text` into active domains and compare against observed `limb_confidences`.
///
/// `limb_confidences` is a map of limb_name → confidence (0.0–1.0).
pub fn analyse(
    text: &str,
    limb_confidences: &HashMap<String, f32>,
) -> DomainAnalysis {
    let lower = text.to_lowercase();

    // ── 1. Detect triggered domains ──────────────────────────────────────────
    let mut active_domains: Vec<DomainHit> = Vec::new();
    let domain_high_map: HashMap<&str, &[&str]> = DOMAIN_HIGH_LIMBS.iter().cloned().collect();

    for &(domain, keywords) in DOMAIN_TRIGGERS {
        let found: Vec<String> = keywords
            .iter()
            .filter(|&&kw| lower.contains(kw))
            .map(|kw| kw.to_string())
            .collect();

        if !found.is_empty() {
            let expected_high: Vec<String> = domain_high_map
                .get(domain)
                .unwrap_or(&&[][..])
                .iter()
                .map(|s| s.to_string())
                .collect();

            active_domains.push(DomainHit {
                domain: domain.to_string(),
                triggers: found,
                expected_high,
            });
        }
    }

    // ── 2. Planning goal detection ───────────────────────────────────────────
    let planning_has_goal = PLANNING_GOAL_KEYWORDS
        .iter()
        .any(|&kw| lower.contains(kw));

    // ── 3. Build per-limb expectations ───────────────────────────────────────
    // Collect which limbs should be high or low
    let mut expected_high: HashMap<String, Vec<String>> = HashMap::new();  // limb → domains
    for hit in &active_domains {
        for limb in &hit.expected_high {
            expected_high
                .entry(limb.clone())
                .or_default()
                .push(hit.domain.clone());
        }
    }

    // All known limbs
    let all_limbs = [
        "memory", "planning", "language", "spatial",
        "reasoning", "metacognition", "action",
        "visualization", "imagination", "empathy", "emotion", "ethics",
    ];

    let high_threshold = 0.60_f32;
    let low_threshold  = 0.45_f32;

    let mut limb_expectations: Vec<LimbExpectation> = Vec::with_capacity(all_limbs.len());
    let mut met_count = 0u32;
    let mut total_count = 0u32;

    for &limb in &all_limbs {
        let actual = *limb_confidences.get(limb).unwrap_or(&0.5);

        // Special case: planning with no concrete goal → expect LOW
        let (expected_label, met) = if limb == "planning" && !planning_has_goal {
            let met = actual < low_threshold;
            ("low", met)
        } else if expected_high.contains_key(limb) {
            let met = actual >= high_threshold;
            ("high", met)
        } else {
            // No strong domain expectation → neutral (always met)
            ("neutral", true)
        };

        if expected_label != "neutral" {
            total_count += 1;
            if met { met_count += 1; }
        }

        limb_expectations.push(LimbExpectation {
            limb: limb.to_string(),
            expected: expected_label,
            actual_confidence: actual,
            met,
        });
    }

    let sensitivity_score = if total_count == 0 {
        1.0
    } else {
        met_count as f32 / total_count as f32
    };

    DomainAnalysis {
        active_domains,
        planning_has_goal,
        limb_expectations,
        sensitivity_score,
    }
}
